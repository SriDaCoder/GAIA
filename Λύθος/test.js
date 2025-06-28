const crypto = require('crypto');
const express = require('express');

app.use(express.static('public'));
// Import necessary libraries
// const crypto = require('crypto');
// const express = require('express');
// const bodyParser = require('body-parser');
// const cors = require('cors');
// const { Buffer } = require('buffer');

const app = express();
app.use(express.json());

// Helper: chaotic logistic map generator
function logisticMap(seed, size, r = 3.99) {
    let x = seed;
    const result = [];
    for (let i = 0; i < size; i++) {
        x = r * x * (1 - x);
        result.push(Math.floor(x * 256) % 256);
    }
    return result;
}

// Helper: 512-bit block padding
function padBlock(data, blockSize = 64) {
    while (data.length < blockSize) {
        data.push(0);
    }
    return data.slice(0, blockSize);
}

// Substitution using chaotic S-box
function substitute(block, sBox) {
    return block.map(b => sBox[b]);
}

// Permutation step
function permute(block, permMap) {
    return permMap.map(i => block[i]);
}

// Round function
function roundEncrypt(block, sBox, noiseMask, shiftAmount) {
    block = substitute(block, sBox);
    block = permute(block, noiseMask.map((_, i) => i).sort((a, b) => noiseMask[a] - noiseMask[b]));
    block = block.map((b, i) => b ^ noiseMask[i]);
    return block.map(b => ((b << shiftAmount) | (b >> (8 - shiftAmount))) & 0xFF);
}

// Main encryption function
function lythosEncrypt(data, key, rounds = 14) {
    if (key.length !== 32) {
        throw new Error("Key must be 256-bit (32 bytes)");
    }

    let block = padBlock(Array.from(data), 64);
    const seed = (key.slice(0, 4).reduce((acc, val) => (acc << 8) | val, 0)) / Math.pow(2, 32);
    let sBox = logisticMap(seed, 256);
    sBox = sBox.map((_, i) => i).sort((a, b) => sBox[a] - sBox[b]); // Chaotic S-box

    let encrypted = block;
    for (let i = 0; i < rounds; i++) {
        const noiseSeed = (parseInt(crypto.createHash('sha256').update(Buffer.concat([key, Buffer.from([i])])).digest('hex').slice(0, 8), 16)) / Math.pow(2, 32);
        const noiseMask = logisticMap(noiseSeed, encrypted.length);
        const shiftAmount = (noiseMask[0] % 7) + 1;
        encrypted = roundEncrypt(encrypted, sBox, noiseMask, shiftAmount);
    }

    // Add tag
    const tag = crypto.createHash('sha256').update(Buffer.concat([key, Buffer.from(encrypted)]))
        .digest().slice(0, 8);
    return Buffer.concat([Buffer.from(encrypted), tag]);
}

app.post('/encrypt', (req, res) => {
    try {
        console.log("Incoming request:", req.body);
        const data = Buffer.from(req.body.data, 'utf-8');
        const key = Buffer.from(req.body.key, 'utf-8');

        const ciphertext = lythosEncrypt(data, key);
        res.json({ ciphertext: ciphertext.toString('hex') });
    } catch (e) {
        console.error("Encryption error:", e.message);
        res.status(400).json({ error: e.message });
    }
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});