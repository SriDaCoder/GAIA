import hashlib

def calculate_file_hash(filepath, hash_function='sha256'):
    hash_func = hashlib.new(hash_function)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

# Compute the initial hash
filepath = '/c:/Users/srini/OneDrive/Desktop/Coding/GAIA/Προμηθεύς/main-test-side.py'
initial_hash = calculate_file_hash(filepath)
print(f"Initial hash: {initial_hash}")