from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
import cv2, numpy as np

class Cam(Camera):
    def build(self):
        pass

    def get_texture(self):
        # print "Texture", self.texture
        return self.texture

class CameraApp(App):
    def build(self):
        s = 100
        size = (s, s)
        sh = (None, None)
        self.snapshot = None
        root = Widget()
        self.width = 640
        self.height = 480
        self.channels = 3
        self.camera = Cam(resolution=(self.width, self.height), size=(self.width, self.height), pos=(100,100), play=True)
        self.display = Widget(size=(self.width, self.height))
        root.add_widget(self.camera)
        root.add_widget(self.display)
        buttons = BoxLayout(orientation='horizontal')
        sn = Button(text='snapshot', pos=(self.width/2, 20), halign='center', size_hint=sh, size=size)
        sn.bind(on_press=self.get_texture1)
        buttons.add_widget(sn)
        root.add_widget(buttons)
        # print "Texture", self.camera.get_texture, type(self.camera.get_texture), dir(self.camera.get_texture)
        Clock.schedule_interval(self.get_texture1, 1.0/33.0)
        return root

    def get_texture1(self, event):
        self.snapshot = self.camera.get_texture()
        # print "self.snapshot", self.snapshot, type(self.snapshot), dir(self.snapshot)
        self.reg = self.snapshot.get_region(0, 0, 50, 50)
        # print "self.reg.pixels", type(self.reg.pixels)
        # print self.reg, type(self.reg), dir(self.reg)
        frame = np.fromstring(self.snapshot.pixels,
                dtype=np.uint8,
                count=self.width*self.height*self.channels)
        frame = frame.reshape(self.height, self.width, self.channels)
        # print dir(frame)
        # print type(frame)
        image = Texture.create(size=frame.shape[:2])
        image = np.rot90(np.swapaxes(image, 0, 1))
        image.blit_buffer(frame.tostring(), colorfmt='rgba', bufferfmt='ubyte')

        with self.display.canvas:
            Rectangle(texture=image, pos=self.display.pos, size=self.display.size)

if __name__ == '__main__':
    CameraApp().run()
