import os, random
os.system('cls' if os.name == 'nt' else 'clear')

origin = input("Enter code: ")
encryption_table = {}
encryption_key = ""
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for each in alphabets:
    random_num = random.randint(1, len(alphabets) + len(origin))
    encryption_key += str(random_num)
    encryption_table[each] = str(random_num)

output = None

def encryptor(code):
    if code != "":
        output = ""
        for i in code:
            if i in encryption_table and random_num % 2 == 0:
                output += str(encryption_table[i])
            elif i in encryption_table and random_num % 2 == 1:
                output += i
            else:
                pass
    return output

if origin != "":
    file_path = input("File: ")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            x = file.readline()
            print("Encrypted text:", encryptor(x))
            print("Encryption Key:", encryption_key)
            x = input("Do you want to save the encrypted text? (y/n): ").strip().lower()
            if x == 'y':
                with open(file_path, 'w') as file:
                    file.write(encryptor(x))
                    print(f"Encrypted text saved to {file_path}.")
            else:
                print("Encrypted text not saved.")
    else:
        print("File does not exist. Please check the file path and try again.")