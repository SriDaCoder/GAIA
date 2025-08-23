from flask import Flask, request, jsonify
import random

app = Flask(__name__)

base40chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ@#$%"

def from_base40(s):
    s = s.upper()
    num = 0
    for ch in s:
        val = base40chars.find(ch)
        if val == -1:
            return None
        num = num * 40 + val
    return str(num)

def to_base40(num_str):
    num = int(num_str)
    if num == 0:
        return "0"
    result = ""
    while num > 0:
        result = base40chars[num % 40] + result
        num //= 40
    return result

def import_key(key_str):
    return [key_str[i:i+3] for i in range(0, len(key_str), 3)]

def get_char_to_code_table(key):
    return {chr(32 + i): key[i] for i in range(len(key))}

def encrypt_text(text, key):
    table = get_char_to_code_table(key)
    return ''.join(table.get(c, c) for c in text)

def generate_key(provided_key=None):
    if provided_key:
        decoded_num_str = from_base40(provided_key)
        if not decoded_num_str:
            return None
        return import_key(decoded_num_str)
    else:
        used = set()
        key = []
        for _ in range(32, 127):
            while True:
                code = str(random.randint(100, 999))
                if code not in used:
                    break
            used.add(code)
            key.append(code)
        return key

@app.route("/encrypt", methods=["POST"])
def encrypt():
    data = request.json
    msg = data.get("message", "")
    provided_key = data.get("key")  # optional Base40 key

    key = generate_key(provided_key)
    if key is None:
        return jsonify({"error": "Invalid key format"}), 400

    encrypted = encrypt_text(msg, key)
    base40key = provided_key if provided_key else to_base40(''.join(key))

    return jsonify({
        "encrypted": encrypted,
        "key": base40key
    })

if __name__ == "__main__":
    app.run(debug=True)
