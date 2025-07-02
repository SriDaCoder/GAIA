import random

# Generate 3-digit code per printable ASCII character (space to ~)
def generate_key():
    chars = [chr(i) for i in range(32, 127)]  # ASCII 32–126
    used = set()
    key = []
    for _ in chars:
        while True:
            code = str(random.randint(100, 999))
            if code not in used:
                used.add(code)
                key.append(code)
                break
    return key  # List of 95 codes in ASCII order

def get_char_to_code_table(key):
    chars = [chr(i) for i in range(32, 127)]
    return dict(zip(chars, key))

def encrypt(text, key):
    table = get_char_to_code_table(key)
    return ''.join(table.get(c, c) for c in text)

def export_key(key):
    return ''.join(key)  # Just 95 * 3 = 285 digits

# Main
if __name__ == "__main__":
    msg = input("Enter text to encrypt: ")
    key_list = generate_key()
    encrypted = encrypt(msg, key_list)
    key_str = export_key(key_list)

    print("\n--- Encrypted ---")
    print(encrypted)

    print("\n--- Key ---")
    print(key_str)
