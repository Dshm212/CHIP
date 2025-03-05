import hashlib
import random

import torch
from Crypto.Util import number


def tensor_hash(tensor):
    tensor_bytes = tensor.cpu().numpy().tobytes()
    hash_object = hashlib.sha512(tensor_bytes)
    hash_bytes = hash_object.digest()
    hash_value = int.from_bytes(hash_bytes, byteorder='big')

    return hash_value


def license_to_integer(license_str, q):
    byte_array = bytearray(license_str, 'utf-8')

    if len(byte_array) > q.bit_length() // 8:
        raise ValueError("License should no longer than", q.bit_length() // 8, " bytes")

    r = int.from_bytes(byte_array, byteorder='big')
    # length_of_byte_array = len(byte_array)

    return r


def recover_license(s):
    byte_length = (s.bit_length() + 7) // 8
    s_bytes = s.to_bytes(byte_length, byteorder='big')

    # print(len(s_bytes))

    try:
        s_recovered_text = s_bytes.decode('utf-8')
        print("Successfully decoded recovered bytes to text.")
        print(f"Recovered text: {s_recovered_text}")
    except UnicodeDecodeError:
        print("Failed to decode recovered bytes to text.")
        print("Trying with latin-1 encoding...")
        s_recovered_text = s_bytes.decode('latin-1')
        print(f"Recovered text with latin-1: {s_recovered_text}")


def keygen(bits, random_seed=None):
    def randfunc(n):
        return random.getrandbits(n * 8).to_bytes(n, 'big')

    # set the random seed
    # if random_seed is not None:
    random.seed(random_seed)

    while True:
        q = number.getPrime(bits, randfunc=randfunc)
        p = 2 * q + 1

        if number.isPrime(p):
            break

    # g = random.randint(1, p - 1)
    # g = pow(g, 2, p)

    while True:
        tmp = random.randint(2, p - 2)
        g = pow(tmp, 2, p)
        if g != 1 and pow(g, q, p) == 1:
            break

    tk = random.randint(1, q)
    hk = pow(g, tk, p)

    return p, q, g, hk, tk


def randgen(q, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    return random.randint(0, q)


def chameleon_hash(hk, p, q, g, message, r):
    e = tensor_hash(message)

    hke = pow(hk, e, p)
    gs = pow(g, r, p)

    hash_value = (hke * gs) % p

    return hash_value


def generate_collision(params, msg1, msg2):
    p, q, g, hk, tk = params

    e1 = tensor_hash(msg1)
    e2 = tensor_hash(msg2)

    r2 = (r1 + (e1 - e2) * tk) % q

    h2 = chameleon_hash(hk, p, q, g, msg2, r2)

    return r2, h2


def owner_chameleon_hash(message, license, hash_length=512):
    p, q, g, hk, tk = keygen(256, random_seed=42)

    r1 = license_to_integer(license, q)

    if hash_length > 512:
        return "Hash length must be less than or equal to 512 bits"

    hash_value_int = chameleon_hash(hk, p, q, g, message, r1)

    # Convert int hash to binary string
    full_binary_hash = bin(hash_value_int)[2:]  # SHA-512 hash is 512 bits

    # Truncate or pad the binary string to the specified length
    if len(full_binary_hash) >= hash_length:
        truncated_hash = full_binary_hash[:hash_length]
    else:
        print("hash overlength", len(full_binary_hash))
        num_repeats = (hash_length + len(full_binary_hash) - 1) // len(full_binary_hash)
        full_binary_hash = full_binary_hash * num_repeats
        truncated_hash = full_binary_hash[:hash_length]

        # truncated_hash = full_binary_hash.ljust(hash_length, '0')

    binary_hash = list(map(int, truncated_hash))
    binary_hash = torch.tensor(binary_hash)
    binary_hash = torch.sign(binary_hash - 0.5)

    try:
        assert len(binary_hash) == hash_length
    except:
        print('Invalid binary hash length for the passport license!, see models/layers/hash.py')
        exit()

    return (p, q, g, hk, tk), r1, hash_value_int, binary_hash


if __name__ == '__main__':
    # msg1 = "YES"
    msg1 = torch.rand((1, 192, 8, 8))
    l1_text = "Copyright to CVPR 2025"
    params, r1, hash1, binary_hash1 = owner_chameleon_hash(msg1, l1_text)

    p, q, g, hk, tk = params

    # check wether p and q are prime or not
    print(number.isPrime(p))
    print(number.isPrime(q))

    print(f"p: {p},\n q: {q},\n g: {g},\n hk: {hk},\n tk: {tk}")

    # msg2 = "NO"
    msg2 = torch.rand((1, 192, 8, 8))
    r2, hash2 = generate_collision(params, msg1, msg2)

    print(f"hash1: {hash1}\nhash2: {hash2}")

    assert hash1 == hash2, "Collision failed!"
    print("Collision successful!")

    recover_license(r1)
    recover_license(r2)
