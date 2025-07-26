import rsa


message = "testing yet another process"

with open('public_key.pem', 'rb') as p:
    public_key = rsa.PublicKey.load_pkcs1(p.read())

def encrypt(message, key):
    return rsa.encrypt(message.encode('ascii'), key)

encryption = encrypt(message, public_key)

with open('encrypted_message.txt', 'wb') as p:
    p.write(encryption)