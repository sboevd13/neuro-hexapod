"""Record actual GUI frames from a scripted W-key trial on each course."""
import argparse
import subprocess
from pathlib import Path
import pygame as pg
from .viewer import Console

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=41)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1000x600',
         '-r','12','-i','-','-an','-c:v','libx264','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)]
    app=Console('flat',args.seed)
    try:
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as writer:
            for name,seconds in [('low',7.),('narrow',10.),('rough',8.),('blocked',2.5)]:
                app.reset(name)
                app.event(pg.event.Event(pg.KEYDOWN,key=pg.K_w))
                for i in range(round(seconds*50)):
                    app.step()
                    if i%10==0:
                        app.draw()
                        writer.stdin.write(pg.image.tobytes(app.canvas,'RGB'))
                app.event(pg.event.Event(pg.KEYUP,key=pg.K_w))
            writer.stdin.close()
            if writer.wait()!=0:
                raise RuntimeError('ffmpeg failed')
    finally:
        app.close()

if __name__=='__main__':
    main()
