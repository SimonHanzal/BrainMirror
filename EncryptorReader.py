import rsa
import numpy as np

with open('private_key.pem', 'rb') as p:
    private_key = rsa.PrivateKey.load_pkcs1(p.read())

def decrypt(ciphertext, key):
    try:
        return rsa.decrypt(ciphertext, key).decode('ascii')
    except:
        return False

with open('encrypted_message.txt', 'rb') as p:
    encryption = p.read()

decryption = decrypt(encryption, private_key)

try:
    print(decryption)
except:
    pass