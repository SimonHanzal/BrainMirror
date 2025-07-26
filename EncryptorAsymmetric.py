import rsa

def generateKeys():
    (public_key, private_key) = rsa.newkeys(512)
    with open('public_key.pem', 'wb') as p:
        p.write(public_key.save_pkcs1('PEM'))
    with open('private_key.pem', 'wb') as p:
        p.write(private_key.save_pkcs1('PEM'))


generateKeys()