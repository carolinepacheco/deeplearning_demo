#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:37:23 2017

@author: carolinepacheco
"""

import argparse
import cv2
import json


# python mainTest.py -i images/anomaly.jpg -o output/prediction.jpg -j json/predictions3.json
# python mainTest.py -i images/normal.jpg -o output/prediction.jpg -j json/predictions2.json

def main():
  parser = argparse.ArgumentParser(description='Anomaly detection')
  parser.add_argument('-i','--img_in', help='Input image', required=True)
  parser.add_argument('-o','--img_out', help='Output image', required=True)
  parser.add_argument('-j','--json_in', help='Json file', required=True)
  
  args = parser.parse_args()
  if args.img_in:
    print("reading %s..." % args.img_in)
  if args.img_out:
    print("output file %s..." % args.img_out)
  if args.json_in:
    print("Json file %s..." % args.json_in) 
   # config = json.loads(open(json_in).read())
  img_in = args.img_in
  img_out = args.img_out
  json_in = args.json_in
  resp = loadData(json_in)
  anomalyDetection(resp, img_in, img_out)
  #showInfo(imgout, j, v, resp)

def loadData(json_in):
    #load the data into an element
    config = json.loads(open(json_in).read())
    #dumps the json object into an element
    json_str = json.dumps(config)
    resp = json.loads(json_str)
    return [resp]

def anomalyDetection(resp, img_in, img_out): 
    print("Searching for anomalies in the scene")
    orig = cv2.imread(img_in)
    i = 1
    j = 0
    v=[]   #anamoly position  
    while i <= len(resp[0]):
        a = i;
        a = str(a)
        x = (resp[0][a])
        var1 = x[0]        
        if var1 != 'person':
          cv2.rectangle(orig,(x[2], x[3]),(x[4], x[5]),(0,0,255),2)
          j += 1
          v.append(i)
        i += 1    
        cv2.imwrite(img_out, orig) 
    
    if j == 0:
           cv2.putText(orig, "Anomaly not detected", (10, 30), 
             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
           cv2.imwrite(img_out, orig) 
    else:
         ii = 0 
         while ii < len(v):
             b = v[ii] 
             b = str(b)
             x1 = (resp[0][b])      
             x3 = (x1[ii])
             x2 = (x1[1])
             ii += 1
             cv2.putText(orig, "Number of anomalies detected: {}".format(j), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
             cv2.putText(orig, "Type of anomaly: {}".format(x3), (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (102, 0, 255), 2)
             cv2.putText(orig, "({0:.0f}%)".format(x2), (330, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (102, 0, 255), 2)         
             ii += 1
             cv2.imwrite(img_out, orig)              

if __name__ == "__main__":
  main()  