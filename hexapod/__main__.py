import argparse
import json
import math
import os
import sys
from pathlib import Path
from .world import COURSES,TITLES

def main():
    p=argparse.ArgumentParser(description='Дистанционное управление гексаподом в MuJoCo.')
    p.add_argument('--track',choices=COURSES,default='mixed')
    p.add_argument('--list-tracks',action='store_true')
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--input',choices=('keyboard','gamepad'),default='keyboard')
    p.add_argument('--replay',type=Path)
    p.add_argument('--controller',choices=('adaptive','baseline'),default='adaptive')
    p.add_argument('--camera',choices=('depth','rgb'),default='depth',
                   help='Metric depth or learned monocular perception. Both operator views show RGB.')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seconds',type=float)
    args=p.parse_args()
    if args.list_tracks:
        for i,(name,title) in enumerate(zip(COURSES,TITLES),1):
            print(f'{i}  {name:10s} {title}')
        return
    if args.input=='gamepad' and (args.headless or args.replay):
        p.error('Headless and replay modes use recorded commands; choose --input keyboard or omit --input')
    if args.seconds is not None and (not args.headless or not math.isfinite(args.seconds) or args.seconds<=0):
        p.error('--seconds requires headless mode and a positive duration')
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
    from .inputs import Replay
    replay=Replay.load(args.replay) if args.replay else None
    if args.headless:
        from .benchmark import run
        result=run(args.track,args.seed,args.controller=='adaptive',args.seconds,source=replay,camera=args.camera)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    from .viewer import Console
    try:
        app=Console(args.track,args.seed,args.input,args.camera,args.controller=='adaptive',replay)
        app.loop()
    except ValueError as exc:
        p.exit(1,f'{exc}\n')

if __name__=='__main__':
    main()
