"""Desktop operator console with two live views and hot-pluggable command sources."""
import time
import mujoco
import pygame as pg
from .controller import Controller
from .inputs import Keyboard,Gamepad
from .ports import DT,tick
from .simulation import Simulation
from .world import CAMERA,COURSES,TITLES

STATUS={'normal':'Обычная походка','low':'Опускаю корпус','narrow':'Сужаю постановку ног',
        'rough':'Выбираю опору на неровностях','blocked':'Проход закрыт','sensor_stop':'Нет измерения'}
BG=(21,28,34)
TEXT=(232,238,235)
MUTED=(170,187,188)
ACCENT=(122,216,186)

class View:
    def __init__(self,sim):
        self.sim=sim
        self.renderer=mujoco.Renderer(sim.model,420,640)
        self.small=mujoco.Renderer(sim.model,144,224)
        self.chase=mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.chase)
        self.chase.type=mujoco.mjtCamera.mjCAMERA_FREE
        self.chase.azimuth=58.
        self.chase.elevation=-23.
        self.chase.distance=1.08

    def frame(self,onboard=False,small=False):
        self.chase.lookat[:]=self.sim.data.qpos[:3]+[.08,0,.015]
        renderer=self.small if small else self.renderer
        renderer.update_scene(self.sim.data,camera=CAMERA if onboard else self.chase)
        return renderer.render()

    def close(self):
        self.small.close()
        self.renderer.close()

class Console:
    def __init__(self,track,seed,input_name='keyboard',camera='depth',adaptive=True,replay=None):
        pg.display.init()
        pg.font.init()
        self.screen=pg.display.set_mode((1000,600),pg.RESIZABLE)
        pg.display.set_caption('Гексапод · MuJoCo')
        self.canvas=pg.Surface((1000,600))
        self.font=pg.font.SysFont('dejavusans,arial',19)
        self.small_font=pg.font.SysFont('dejavusans,arial',15)
        self.title_font=pg.font.SysFont('dejavusans,arial',27,bold=True)
        self.keyboard=Keyboard()
        self.source=self.keyboard
        self.input_name=input_name
        self.gamepad=None
        if input_name=='gamepad':
            pg.joystick.init()
            if pg.joystick.get_count()<1:
                pg.quit()
                raise ValueError('Геймпад не подключён. Доступен запуск с --input keyboard.')
            self.attach_pad(0)
        if replay is not None:
            self.source=replay
        self.seed,self.camera,self.adaptive=seed,camera,adaptive
        self.onboard=False
        self.running=True
        self.view=self.sim=None
        self.reset(track)

    def attach_pad(self,index):
        self.gamepad=pg.joystick.Joystick(index)
        self.gamepad.init()
        self.source=Gamepad(self.gamepad)

    def reset(self,track):
        if self.view:
            self.view.close()
        if self.sim:
            self.sim.close()
        self.track=track
        self.sim=Simulation(track,self.seed,self.camera)
        self.view=View(self.sim)
        self.ctrl=Controller(self.adaptive)
        self.keyboard.held.clear()
        self.paused=False
        self.mode='normal'

    def event(self,event):
        if event.type==pg.QUIT:
            self.running=False
        elif event.type==pg.WINDOWFOCUSLOST:
            self.keyboard.held.clear()
        elif self.input_name=='gamepad' and event.type==pg.JOYDEVICEREMOVED:
            if self.gamepad and self.gamepad.get_instance_id()==event.instance_id:
                self.gamepad=None
                self.keyboard.held.clear()
                self.source=self.keyboard
        elif self.input_name=='gamepad' and event.type==pg.JOYDEVICEADDED and self.gamepad is None:
            self.attach_pad(event.device_index)
        elif event.type in (pg.KEYDOWN,pg.KEYUP):
            down=event.type==pg.KEYDOWN
            key=pg.key.name(event.key)
            if down and not getattr(event,'repeat',False):
                if key=='escape':
                    self.running=False
                elif key=='tab':
                    self.onboard=not self.onboard
                elif key=='r':
                    self.reset(self.track)
                elif key in tuple(str(i+1) for i in range(len(COURSES))):
                    self.reset(COURSES[int(key)-1])
                elif key=='space':
                    self.paused=not self.paused
                    self.keyboard.held.clear()
            if self.source is self.keyboard:
                self.keyboard.key(key,down)

    def step(self):
        if not self.paused:
            self.mode=tick(self.source,self.sim,self.ctrl,self.sim).mode

    def label(self,text,pos,font=None,color=TEXT):
        self.canvas.blit((font or self.font).render(text,True,color),pos)

    def picture(self,array,rect):
        surface=pg.image.frombuffer(array.tobytes(),(array.shape[1],array.shape[0]),'RGB')
        self.canvas.blit(pg.transform.smoothscale(surface,rect.size),rect)

    def draw(self):
        self.canvas.fill(BG)
        self.label(TITLES[COURSES.index(self.track)],(20,10),self.title_font)
        self.label('С камеры' if self.onboard else 'Вид со стороны',(22,51),self.small_font,MUTED)
        self.picture(self.view.frame(self.onboard),pg.Rect(20,80,665,436))
        self.label('Вид со стороны' if self.onboard else 'Бортовая камера',(704,51),self.small_font,MUTED)
        self.picture(self.view.frame(not self.onboard,True),pg.Rect(704,80,276,178))
        self.label('Режим движения',(704,284),self.small_font,MUTED)
        notice='Пауза' if self.paused else STATUS[self.mode]
        self.label(notice,(704,313),self.small_font if len(notice)>24 else self.font,
                   (251,180,111) if self.mode in ('blocked','sensor_stop') else ACCENT)
        if self.mode in ('blocked','sensor_stop'):
            self.label('S — отступить',(704,344),self.small_font)
        self.label('W / S   вперёд / назад',(704,397),self.small_font)
        self.label('A / D   влево / вправо',(704,425),self.small_font)
        self.label('Q / E   поворот',(704,453),self.small_font)
        self.label('1–6   трасса      Tab   камера      R   сначала      Space   пауза      Esc   выход',
                   (20,551),self.small_font,MUTED)
        self.label(f'{self.sim.time:.1f} с',(880,549),self.small_font,MUTED)
        self.screen.blit(pg.transform.smoothscale(self.canvas,self.screen.get_size()),(0,0))
        pg.display.flip()

    def loop(self):
        last=time.monotonic()
        balance=0.
        clock=pg.time.Clock()
        try:
            while self.running:
                for event in pg.event.get():
                    self.event(event)
                now=time.monotonic()
                balance+=min(.15,now-last)
                last=now
                while balance>=DT:
                    self.step()
                    balance-=DT
                self.draw()
                clock.tick(30)
        finally:
            self.close()

    def close(self):
        if self.view:
            self.view.close()
            self.view=None
        if self.sim:
            self.sim.close()
            self.sim=None
        pg.quit()
