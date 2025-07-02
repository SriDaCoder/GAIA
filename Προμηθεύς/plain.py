def get_decryption_table(key):
    # Reverse of encryption table: number → letter
    table = {}
    for i, letter in enumerate(string.ascii_lowercase):
        num = key[i*3:i*3+3]
        table[num] = letter
    return table

def decrypt(text, key):
    table = get_decryption_table(key)
    result = ""
    i = 0
    while i < len(text):
        chunk = text[i:i+3]
        if chunk in table:
            result += table[chunk]
            i += 3
        else:
            result += text[i]
            i += 1
    return result

if __name__ == "__main__":
    encrypted_text = input("Enter text to decrypt: ")
    key = input("Enter the key used for encryption: ")
    decrypted_text = decrypt(encrypted_text, key)
    print("Decrypted:", decrypted_text)