import os,json,time,image,gc
try:os.mkdir('/flash/research-v5-lbp')
except OSError:pass
for seed in range(10):
    frame=image.Image(91,91,image.GRAYSCALE);raw=frame.bytearray()
    for y in range(91):
        for x in range(91):raw[y*91+x]=(x*17+y*31+seed*23+(x*y)%29)%256
    start=time.ticks_us();desc=frame.find_lbp();duration=time.ticks_diff(time.ticks_us(),start)
    image.save_descriptor('/flash/research-v5-lbp/pattern_%02d.lbp'%seed,desc)
    self_distance=image.match_descriptor(desc,desc)
    print(json.dumps({'seed':seed,'descriptor_us':duration,'self_distance':self_distance,'kind':'lbp_native','shape':[91,91]}))
    desc=None;raw=None;frame=None;gc.collect()
print(json.dumps({'kind':'complete'}))
