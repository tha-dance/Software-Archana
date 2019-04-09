import pandas as pd
from feature_extraction import extract
from sklearn.externals import joblib
import numpy as np


import serial
import collections
import time
import os
import socket
import base64
import sys
from Crypto.Cipher import AES
from Crypto import Random

port = "/dev/ttyS0"
s1 = serial.Serial(port, baudrate=115200)

bufferSize = 50

val = 0
handshake = 0
buffer = []
checksumPi = 0
duinoConnectionEstablished = False
pkt = 0

VERBOSE = False
sock = None
serverConnectionEstablished = False
paddedData = ""
Key = 'please give us A'
secret_key = bytes(str(Key), encoding = "utf8")

#Asumptions Voltage & Current 1dp




moves = {
1: 'Raffles',
2: 'Chicken',
3: 'Crab',
4: 'Hunchback',
5: 'Cowboy'
}
scaler = joblib.load('scaler.joblib')
rf = joblib.load('rf_final.joblib')
mlp = joblib.load('mlp_final.joblib')


def form_segment(data, segment):
    window_size = 50
    segment.extend(data)
    if (len(segment)== window_size):
        return True, segment
    else:
        return False, segment

def extract_feature(segment):
    data = np.asarray(extract(np.asarray(segment)))
    #data = np.array([data])
    data = scaler.transform(data)
    return data

#DATA = []
#PREDS = []

def MLstuff(segment):
    #isSegment, segment = form_segment(data,segment)

    #if (isSegment):
        # if segment has been formed
        extracted_features = extract_features(segment)
        rf_pred = int(rf.predict(extracted_features))
        mlp_pred = int(mlp.predict(extracted_features))
        pred_arr = []
        #PREDS.extend((rf_pred, mlp_pred))
        #if len(PREDS) == 3:

        pred_arr.append(rf_pred)
        pred_arr.append(mlp_pred)

        #segment = segment[25:]
        mode, num_mode = Counter(pred_arr).most_common(1)[0]
        if (num_mode>=2):
        	final_pred = moves.get(mode)
        	return final_pred

    return 

            

def calcCheckSum(arr, volt, curr):

    checksum = arr[0] ^ arr[1]
    for i in range(2, len(arr)) :
        checksum = checksum ^ arr[i]
    checksum = checksum ^ int(volt*10)
    checksum = checksum ^ int(curr*10)
    return checksum

def createReadingArr(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    arr = []
    arr.append(x1)
    arr.append(y1)
    arr.append(z1)
    arr.append(x2)
    arr.append(y2)
    arr.append(z2)
    arr.append(x3)
    arr.append(y3)
    arr.append(z3)
    return arr

def debug(text):
    if VERBOSE:
        print ("Debug:---", text)

def sendMSG(msg):
    debug("sendMSG() with msg")
    try:
        sock.sendall(msg)
    except:
        debug("Exception in sendMSG()")
        closeConnection()

def closeConnection():
    global serverConnectionEstablished
    debug("Closing socket")
    sock.close()
    serverConnectionEstablished = False

def connect(IP_ADDRESS, IP_PORT):
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    debug("Connecting to Server...")
    try:
        sock.connect((IP_ADDRESS, IP_PORT))
    except:
        debug("Connection to Server failed.")
        return False
    return True

def formStr(act, volt, curr, pow, cumPow):
    return '#' + str(act) + '|' + str(volt) + '|' + str(curr) + '|' + str(pow) + '|' + str(cumPow) + '|'

def encodeStr(data):
    extra = len(data) % 16
    if extra > 0 :
        paddedData = (' ' * (16 - extra)) + data
        
        iv = Random.new().read(AES.block_size) # GENERATING INITIAL VECTOR
        cipher = AES.new(secret_key,AES.MODE_CBC,iv) # CREATING CIPHER
        encryptedMSG = iv + cipher.encrypt(paddedData)
        return base64.b64encode(encryptedMSG)


if len(sys.argv) != 4 :
    print('Invalid number of arguments')
    print('python3 myclient.py [IP address] [Port] [groupID]')
    sys.exit()

IP_ADDRESS = sys.argv[1]
IP_PORT = int(sys.argv[2])
groupID = sys.argv[3]

s1.flushInput()

#HANDSHAKE
while handshake == 0 :
    s1.write(b'0')
    print("Sent request")
    time.sleep(1)
    if s1.in_waiting > 0 :
        val = s1.read(1).decode("utf-8")
        if val == '1' :
            print("Received ACK: " + val)
            handshake = 1
while handshake == 1 :
    s1.write(b'1')
    print("Sent ACK of ACK")
    time.sleep(1)
    if s1.in_waiting > 0 :
        handshake = 2
        duinoConnectionEstablished = True
        print("handshake=2")

s1.flushInput()

#START MAIN
if duinoConnectionEstablished and connect(IP_ADDRESS,IP_PORT):
    serverConnectionEstablished = True
    print ("Connection to Server established")
    time.sleep(16) # 1 second more than server?
    print("going into loop now")
#    while True:
#        time.sleep(1)
#        print("num of bytes in buffer = " + str(s1.in_waiting))
#        if s1.in_waiting :
#            val = s1.read_until().decode("utf-8")# , "ignore")
#            print(val)

    while serverConnectionEstablished:
        while len(buffer) < bufferSize :
            if s1.in_waiting :
                try:
                    val = s1.read_until().decode("utf-8")
                except:
                    print("SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED")
                    val = s1.read_until().decode("utf-8", "ignore")
                    continue
                val = val[:-1]
                print("msg = " + val)

                if val.startswith("#") :
                    val = val.lstrip('#')
                    try:
                        sensor1 = val.split('|')[0]
                        sensor1x = int(sensor1.split(',')[0])
                        sensor1y = int(sensor1.split(',')[1])
                        sensor1z = int(sensor1.split(',')[2])
                        sensor2 = val.split('|')[1]
                        sensor2x = int(sensor2.split(',')[0])
                        sensor2y = int(sensor2.split(',')[1])
                        sensor2z = int(sensor2.split(',')[2])
                        sensor3 = val.split('|')[2]
                        sensor3x = int(sensor3.split(',')[0])
                        sensor3y = int(sensor3.split(',')[1])
                        sensor3z = int(sensor3.split(',')[2])
                        voltage = float(val.split('|')[3])
                        current = float(val.split('|')[4])
                        checksumDuino = int(val.split('|')[5])
                    except ValueError:
                        print("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR")
                        continue
                    print("voltage = " + str(voltage))
                    print("current = " + str(current))
                    print("checksumDuino = " + str(checksumDuino))

                    dataIntArr = createReadingArr(sensor1x, sensor1y, sensor1z, sensor2x, sensor2y, sensor2z, sensor3x, sensor3y, sensor3z)

                    checksumPi = calcCheckSum(dataIntArr, voltage, current)
                    print("checksumPi = " + str(checksumPi))

                    if checksumPi == checksumDuino :
                        print("storing pkt " + str(pkt) + " in buffer")
                        buffer.append(dataIntArr)
                        pkt+=1
        #print (buffer)

        action = MLstuff(buffer)
        if action == 'another segment please' :
            buffer = buffer[25:]
            continue
        
        power = current*voltage
        cumPower = 6.6
        string = formStr(action, voltage, current, power, cumPower)
        print("string = " + string)
        
        encodedMSG = encodeStr(string)
        print ("Sending action: %s" % action)
        sendMSG(encodedMSG)

        time.sleep(1)
        buffer = []
        s1.flushInput()

else:
    print ("Connection to %s:%d failed" % (IP_ADDRESS, IP_PORT))
print ("done")
