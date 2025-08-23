from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Data contracts
# -----------------------------
@dataclass
class Observation:
    soil: torch.Tensor
    water: torch.Tensor
    weather: torch.Tensor

@dataclass
class SubsystemAction:
    command: torch.Tensor
    priority: torch.Tensor
    anomaly: torch.Tensor

@dataclass
class GaiaDecision:
    chosen: torch.Tensor
    provenance: torch.Tensor
    risk: torch.Tensor

# -----------------------------
# Building blocks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int,...]=(128,128), act=nn.ReLU, dropout: float=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), act()]
            if dropout>0:
                layers += [nn.Dropout(dropout)]
            d=h
        layers += [nn.Linear(d,out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Subsystems
# -----------------------------
class DemeterNet(nn.Module):
    def __init__(self, f_soil:int, f_weather:int, action_dim:int):
        super().__init__()
        self.backbone = MLP(f_soil+f_weather,256,(128,128))
        self.head_cmd = MLP(256,action_dim,(128,))
        self.head_pri = MLP(256,1,(64,))
        self.head_anm = MLP(256,1,(64,))
    def forward(self, soil:torch.Tensor, weather:torch.Tensor)->SubsystemAction:
        z = self.backbone(torch.cat([soil,weather],dim=-1))
        return SubsystemAction(
            command=self.head_cmd(z),
            priority=torch.sigmoid(self.head_pri(z)),
            anomaly=torch.sigmoid(self.head_anm(z))
        )

class PoseidonNet(nn.Module):
    def __init__(self, f_water:int, f_weather:int, action_dim:int):
        super().__init__()
        self.backbone = MLP(f_water+f_weather,256,(128,128))
        self.head_cmd = MLP(256,action_dim,(128,))
        self.head_pri = MLP(256,1,(64,))
        self.head_anm = MLP(256,1,(64,))
    def forward(self, water:torch.Tensor, weather:torch.Tensor)->SubsystemAction:
        z = self.backbone(torch.cat([water,weather],dim=-1))
        return SubsystemAction(
            command=self.head_cmd(z),
            priority=torch.sigmoid(self.head_pri(z)),
            anomaly=torch.sigmoid(self.head_anm(z))
        )

class AetherNet(nn.Module):
    def __init__(self, action_dim:int, f_context:int):
        super().__init__()
        self.backbone = MLP(action_dim+f_context,128,(128,64))
        self.head_risk = MLP(128,1,(64,))
        self.head_veto = MLP(128,action_dim,(64,))
    def forward(self, proposed:torch.Tensor, ctx:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        z = self.backbone(torch.cat([proposed,ctx],dim=-1))
        return torch.sigmoid(self.head_risk(z)), torch.sigmoid(self.head_veto(z))

# -----------------------------
# GAIA orchestrator
# -----------------------------
class Gaia(nn.Module):
    def __init__(self, f_soil:int, f_water:int, f_weather:int, action_dim:int):
        super().__init__()
        self.demeter = DemeterNet(f_soil,f_weather,action_dim)
        self.poseidon = PoseidonNet(f_water,f_weather,action_dim)
        self.ctx_fuser = MLP(f_soil+f_water+f_weather,128,(128,))
        self.aether = AetherNet(action_dim,128)
        self.action_dim = action_dim
    def forward(self, obs:Observation)->Tuple[GaiaDecision,Dict[str,SubsystemAction]]:
        dem = self.demeter(obs.soil, obs.weather)
        pos = self.poseidon(obs.water, obs.weather)
        util_dem = dem.priority*(1-dem.anomaly)
        util_pos = pos.priority*(1-pos.anomaly)
        pick_dem = (util_dem>=util_pos).float()
        pick_pos = 1.0 - pick_dem
        proposed = pick_dem*dem.command + pick_pos*pos.command
        provenance = torch.cat([pick_dem,pick_pos,torch.zeros_like(pick_dem)],dim=-1)
        ctx = self.ctx_fuser(torch.cat([obs.soil,obs.water,obs.weather],dim=-1))
        risk,veto = self.aether(proposed,ctx)
        safe_action = proposed*(1.0-(veto>0.5).float())
        decision = GaiaDecision(chosen=safe_action, provenance=provenance, risk=risk)
        return decision, {"DEMETER":dem,"POSEIDON":pos}

# -----------------------------
# Synthetic environment
# -----------------------------
class SyntheticEnv:
    def __init__(self,f_soil=16,f_water=16,f_weather=8,action_dim=6):
        self.f_soil,self.f_water,self.f_weather,self.action_dim=f_soil,f_water,f_weather,action_dim
    def sample_batch(self,batch_size:int,device)->Tuple[Observation,Dict[str,torch.Tensor]]:
        soil = torch.randn(batch_size,self.f_soil,device=device)
        water = torch.randn(batch_size,self.f_water,device=device)
        weather = torch.randn(batch_size,self.f_weather,device=device)
        gt = torch.zeros(batch_size,self.action_dim,device=device)
        cond_dem = (soil.sum(dim=1)<0).float().unsqueeze(-1)
        cond_pos = (water.sum(dim=1)>0).float().unsqueeze(-1)
        gt[:,0]=cond_dem.squeeze()
        gt[:,1]=cond_pos.squeeze()
        storm = (weather.sum(dim=1)>1).float().unsqueeze(-1)
        risk = torch.clamp(gt.abs().sum(dim=1,keepdim=True)*storm,0,1)
        return Observation(soil,water,weather), {"action":gt,"risk":risk}

# -----------------------------
# Loss
# -----------------------------
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
    def forward(self,outputs:Tuple[GaiaDecision,Dict[str,SubsystemAction]],labels:Dict[str,torch.Tensor]):
        decision, subs = outputs
        gt_action = labels["action"]
        gt_risk = labels["risk"]
        dem = subs["DEMETER"]
        pos = subs["POSEIDON"]
        loss_dem_cmd = self.mse(dem.command,gt_action)
        loss_pos_cmd = self.mse(pos.command,gt_action)
        with torch.no_grad():
            dem_match = (gt_action[:,0:1]>0.5).float()
            pos_match = (gt_action[:,1:2]>0.5).float()
        loss_dem_pri = self.mse(dem.priority,dem_match)
        loss_pos_pri = self.mse(pos.priority,pos_match)
        loss_dem_anm = self.mse(dem.anomaly,1-dem_match)
        loss_pos_anm = self.mse(pos.anomaly,1-pos_match)
        loss_aether_risk = self.mse(decision.risk,gt_risk)
        loss_gaia_action = self.mse(decision.chosen,gt_action)
        loss = (loss_dem_cmd + loss_pos_cmd + loss_dem_pri + loss_pos_pri +
                loss_dem_anm + loss_pos_anm + loss_aether_risk + loss_gaia_action)
        return loss, {
            "loss":loss.detach(),
            "dem_cmd":loss_dem_cmd.detach(),
            "pos_cmd":loss_pos_cmd.detach(),
            "aether":loss_aether_risk.detach(),
            "gaia":loss_gaia_action.detach()
        }

# -----------------------------
# Single function to run GAIA
# -----------------------------
def run_gaia(epochs=5, batch_size=8, lr=3e-4, device=torch.device("cpu")):
    env = SyntheticEnv()
    model = Gaia(env.f_soil, env.f_water, env.f_weather, env.action_dim).to(device)
    crit = MultiTaskLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs+1):
        obs, labels = env.sample_batch(batch_size=batch_size, device=device)
        model.train()
        decision, subs = model(obs)
        loss, logs = crit((decision, subs), labels)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        model.eval()
        history.append({k: v.item() for k, v in logs.items()})

    # Return last batch inference for convenience
    return {
        "chosen_action": decision.chosen.detach(),
        "provenance": decision.provenance.detach(),
        "risk": decision.risk.detach(),
        "training_history": history
    }

# -----------------------------
# Run
# -----------------------------
# if __name__ == "__main__":
#    result = run_gaia(epochs=10, batch_size=16)
#    print("Chosen action:", result["chosen_action"])
#    print("Provenance:", result["provenance"])
#    print("Risk:", result["risk"])
#    print("Training history:", result["training_history"])