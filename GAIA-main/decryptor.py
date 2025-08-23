from json import jsonify
import random

class decryptor():
    def __init__(self, text, key=None):
        self.key = key
        self.text = text
        self.base40chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ@#$%"

    def to_base40(self, num_str):
        num = int(num_str)
        if num == 0:
            return "0"
        result = ""
        while num > 0:
            result = self.base40chars[num % 40] + result
            num //= 40
        return result

    def import_key(self, key_str):
        return [key_str[i:i+3] for i in range(0, len(key_str), 3)]

    def get_char_to_code_table(self, key):
        return {chr(32 + i): key[i] for i in range(len(key))}

    def encrypt_text(self, text, key):
        table = self.get_char_to_code_table(key)
        return ''.join(table.get(c, c) for c in text)

    def from_base40(self, s):
        s = s.upper()
        num = 0
        for ch in s:
            val = self.base40chars.find(ch)
            if val == -1:
                return None
            num = num * 40 + val
        return str(num)

    def generate_key(self, provided_key=None):
        if provided_key:
            decoded_num_str = self.from_base40(provided_key)
            if not decoded_num_str:
                return None
            return self.import_key(decoded_num_str)
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

    def encrypt(self):
        msg = self.text
        provided_key = self.key  # optional Base40 key

        key = self.generate_key(provided_key)
        if key is None:
            return jsonify({"error": "Invalid key format"}), 400

        encrypted = self.encrypt_text(msg, key)
        base40key = provided_key if provided_key else self.to_base40(''.join(key))

        return jsonify({
            "encrypted": encrypted,
            "key": base40key
        })