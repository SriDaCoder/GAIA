import hashlib, numpy as np, flask

app = flask.Flask(__name__)

# Helper: chaotic logistic map generator
def logistic_map(seed, size, r=3.99):
    x = seed
    result = []
    for _ in range(size):
        x = r * x * (1 - x)
        result.append(int(x * 256) % 256)
    return result

# Helper: 512-bit block padding
def pad_block(data, block_size=64):
    while len(data) < block_size:
        data += b'\x00'
    return data[:block_size]

# Substitution using chaotic S-box
def substitute(block, s_box):
    return bytes([s_box[b] for b in block])

# Permutation step
def permute(block, perm_map):
    return bytes([block[i] for i in perm_map])

# Round function
def round_encrypt(block, s_box, noise_mask, shift_amount):
    block = substitute(block, s_box)
    block = permute(block, np.argsort(noise_mask))
    block = bytes([(b ^ noise_mask[i]) for i, b in enumerate(block)])
    return bytes([(b << shift_amount | b >> (8 - shift_amount)) & 0xFF for b in block])

# Main encryption function
def lythos_encrypt(data, key, rounds=14):
    if len(key) != 32:
        raise ValueError("Key must be 256-bit (32 bytes)")
    
    block = pad_block(data)
    seed = int.from_bytes(key[:4], 'big') / 2**32
    s_box = logistic_map(seed, 256)
    s_box = np.argsort(s_box)  # Chaotic S-box

    encrypted = block
    for i in range(rounds):
        noise_seed = int.from_bytes(hashlib.sha256(key + i.to_bytes(1, 'big')).digest()[:4], 'big') / 2**32
        noise_mask = logistic_map(noise_seed, len(encrypted))
        shift_amount = (noise_mask[0] % 7) + 1
        encrypted = round_encrypt(encrypted, s_box, noise_mask, shift_amount)

    # Add tag
    tag = hashlib.sha256(key + encrypted).digest()[:8]
    return encrypted + tag

@app.route('/encrypt', methods=['POST'])
def encrypt():
    try:
        data = flask.request.json['data'].encode('utf-8')
        key = flask.request.json['key'].encode('utf-8')
        ciphertext = lythos_encrypt(data, key)
        return flask.jsonify(ciphertext.hex())
    except Exception as e:
        return flask.jsonify(str(e)), 400

if __name__ == "__main__":
    app.run(debug=True)