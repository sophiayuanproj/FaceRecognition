import image,json,binascii,gc
from ml.preprocessing import Normalization
for seed in range(10):
 frame=image.Image(137,113,image.RGB565);raw=frame.bytearray()
 for y in range(113):
  for x in range(137):
   pixel=(((x+seed)%32)<<11)|(((2*y+seed)%64)<<5)|((x*y+seed)%32);i=(y*137+x)*2;raw[i]=pixel&255;raw[i+1]=pixel>>8
 roi=(seed%5,seed%7,127-seed,103-seed);buf=bytearray(64*64*3);Normalization(roi=roi)(frame)(buf,(1,64,64,3),ord('B'))
 print('RGBFIX '+str(seed)+' '+binascii.b2a_base64(buf).decode().strip())
 meta={'kind':'rgb_fixture','seed':seed,'roi':roi,'first_pixel':frame.get_pixel((0,0))}
 if seed==0:meta['r5_to8']=[frame.get_pixel((x,0))[0] for x in range(32)];meta['g6_even_to8']=[frame.get_pixel((0,y))[1] for y in range(32)]
 if seed==1:meta['g6_odd_to8']=[frame.get_pixel((0,y))[1] for y in range(32)]
 print(json.dumps(meta));buf=None;raw=None;frame=None;gc.collect()
print(json.dumps({'kind':'complete','fixtures':10}))
