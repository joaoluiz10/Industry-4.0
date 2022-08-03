#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pyglet

# Comprimento de cada braço, a soma de todos os compeimentos não pode ultrapassar a metade da área 
# de trabalho do robô "comp_trabalho", pois o robô fica no meio.
comp_braco1 = 70  # comprimento do braço da origem
comp_braco2 = 70  # comprimento do braço intermediário
comp_braco3 = 40  # comprimento do braço final ou garra

# Coordenadas do objetivo, só funciona se não estiber randomico e não for mudado pela função ajust_goal()
x_obj = 100. #100.
y_obj = 100. #100
l_obj = 20 # largura

rate_dt = .1 #taxa de refresh, quanto menor, mais lenta a animação

origem_a1_x = 200. # a1 start (x0, y0)
origem_a1_y = 200. # a1 start (x0, y0)
comp_trabalho = 400 # dimensão da área de trabalho quadrada "comp_trabalho x comp_trabalho"
action_dimension = 3
state_dimension = 13

#Parameters
o_a1_x = int(origem_a1_x)
o_a1_y = int(origem_a1_y)
o_a1 = comp_trabalho

class ArmEnv(object):
    viewer = None
    dt = rate_dt    # refresh rate
    action_bound = [-1, 1]
    goal = {'x': x_obj, 'y': y_obj, 'l': l_obj}
    state_dim = state_dimension
    action_dim = action_dimension
    vezes_no_alvo = 10
    
    def ajust_goal(self, x_o, y_o, l_o):
        self.goal['x'] = x_o
        self.goal['y'] = y_o  
        self.goal['l'] = l_o

    def ajust_init_position(self, ang_0, ang_1, ang_2):
        self.arm_info['r'] = (ang_0, ang_1, ang_2)
	
    def vezes_alvo(self, num_alvo):
        self.vezes_no_alvo = num_alvo
		
    def __init__(self):
        self.arm_info = np.zeros(
            action_dimension, dtype=[('l', np.float32), ('r', np.float32)]) # 
        self.arm_info['l'] = comp_braco1, comp_braco2, comp_braco3        # 2 arms length
        self.arm_info['r'] = np.pi/6    # 2 angles information
        self.on_goal = 0

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt # o dt faz a animação fluir continuamente, e não em estados
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([origem_a1_x, origem_a1_y])    # a1 start (x0, y0)  // localização do início do braço 1
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1) // localização do final do braço 1
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2) // localização do final do braço 2
        finger_2 = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # a3 end (x3, y3) // localização do final do braço 3
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / comp_trabalho, (self.goal['y'] - a1xy_[1]) / comp_trabalho]
        dist2 = [(self.goal['x'] - a2xy_[0]) / comp_trabalho, (self.goal['y'] - a2xy_[1]) / comp_trabalho]
        dist3 = [(self.goal['x'] - finger_2[0]) / comp_trabalho, (self.goal['y'] - finger_2[1]) / comp_trabalho]
        r = -np.sqrt(dist3[0]**2+dist3[1]**2) #r = -np.sqrt(dist3[0]**2+dist3[1]**2) - (a1r*3 + a2r*2 + a3r)*2 # penalidade, recompensa negativa
		

        # done and reward
        if (self.goal['x'] - self.goal['l']/2 < finger_2[0] < self.goal['x'] + self.goal['l']/2
        ) and (self.goal['y'] - self.goal['l']/2 < finger_2[1] < self.goal['y'] + self.goal['l']/2):
            r += 1.  # recompensa positiva
            self.on_goal += 1
            if self.on_goal > self.vezes_no_alvo:
                done = True
                a1r_graus = a1r*180/np.pi
                a2r_graus = a2r*180/np.pi
                a3r_graus = a3r*180/np.pi
                print('angulo 1: %f ,  angulo 2: %f , angulo 3: %f  em graus' %(a1r_graus, a2r_graus, a3r_graus))
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/o_a1, a2xy_/o_a1, finger_2/o_a1, dist1 + dist2 + dist3, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand()*comp_trabalho
        self.goal['y'] = np.random.rand()*comp_trabalho
        self.arm_info['r'] = 2 * np.pi * np.random.rand(action_dimension)
        self.on_goal = 0
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, an
        a1xy = np.array([origem_a1_x, origem_a1_y])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        finger_2 = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # a3 end (x3, y3)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/comp_trabalho, (self.goal['y'] - a1xy_[1])/comp_trabalho]
        dist2 = [(self.goal['x'] - a2xy_[0])/comp_trabalho, (self.goal['y'] - a2xy_[1])/comp_trabalho]
        dist3 = [(self.goal['x'] - finger_2[0])/comp_trabalho, (self.goal['y'] - finger_2[1])/comp_trabalho]
        # state
        s = np.concatenate((a1xy_/o_a1, a2xy_/o_a1, finger_2/o_a1, dist1 + dist2 + dist3, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=comp_trabalho, height=comp_trabalho, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.goal_update = goal
        self.center_coord = np.array([o_a1_x, o_a1_y])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), 
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), 
            ('c3B', (249, 86, 86) * 4,))    # color

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
	    # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)
	
	
		# update arm
        (a1l, a2l, a3l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']     # angulos em radianos, isso é o que importa para nosso robô
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # localização final de a1 e inicio a2 (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # localização final de a2 (x2, y2)
        finger_2 = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # a3 end (x3, y3)

        a1tr, a2tr, a3tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - (self.arm_info['r'][0] + self.arm_info['r'][1]), np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        
        xy21_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy22_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy31 = finger_2 + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy32 = finger_2 + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc

        gx = self.goal_update['x'] - self.goal_update['l'] / 2, self.goal_update['y'] - self.goal_update['l'] / 2
        gy = self.goal_update['x'] - self.goal_update['l'] / 2, self.goal_update['y'] + self.goal_update['l'] / 2
        gz = self.goal_update['x'] + self.goal_update['l'] / 2, self.goal_update['y'] + self.goal_update['l'] / 2
        gh = self.goal_update['x'] + self.goal_update['l'] / 2, self.goal_update['y'] - self.goal_update['l'] / 2
        
        
        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))
        self.goal.vertices = np.concatenate((gx, gy, gz, gh))

    #convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())

