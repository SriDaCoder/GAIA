def decrypt(cipher, key):
    table = get_code_to_char_table(key)
    result = ""
    i = 0
    while i < len(cipher):
        chunk = cipher[i:i+3]
        if chunk in table:
            result += table[chunk]
            i += 3
        else:
            result += cipher[i]
            i += 1
    return result

def import_key(key_str):
    return [key_str[i:i+3] for i in range(0, len(key_str), 3)]

def get_code_to_char_table(key):
    chars = [chr(i) for i in range(32, 127)]
    return dict(zip(key, chars))

if __name__ == "__main__":
    # Decrypt
    loaded_key = import_key(input("Key: "))
    decrypted = decrypt(input("Plaintext: "), loaded_key)

    print("\n--- Decrypted ---")
    print(decrypted)