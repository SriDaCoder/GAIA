import random
import string

def generate_key():
    # Create a random 3-digit number for each letter a–z
    alphabet = string.ascii_lowercase
    used_numbers = set()
    key = ""
    for letter in alphabet:
        while True:
            num = str(random.randint(100, 999))
            if num not in used_numbers:
                used_numbers.add(num)
                key += num
                break
    return key  # 26 * 3 = 78-digit key

def get_encryption_table(key):
    # Map each letter to a 3-digit number from the key
    table = {}
    for i, letter in enumerate(string.ascii_lowercase):
        table[letter] = key[i*3:i*3+3]
    return table

def encrypt(text, key):
    table = get_encryption_table(key)
    result = ""
    for char in text.lower():
        if char in table:
            result += table[char]
        else:
            result += char  # Leave symbols, digits, etc.
    return result

if __name__ == "__main__":
    text = input("Enter text to encrypt: ")
    key = generate_key()
    print(f"Generated Key: {key}")
    encrypted_text = encrypt(text, key)
    print("Encrypted:", encrypted_text)