from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys, math, random, time

# -------------------- Global state --------------------
window_width, window_height = 1200, 800

# Crane parameters
boom_angle = 30.0
boom_length = 8.0
mast_height = 10.0
hook_length = 3.0
min_hook, max_hook = 0.5, 15.0
min_boom_angle, max_boom_angle = -10.0, 75.0

# Camera spherical coords
cam_azimuth, cam_elevation, cam_distance = 45.0, 25.0, 35.0

# Steps
angle_step, hook_step, cam_step, zoom_step = 1.5, 0.3, 3.0, 1.0

# Mouse control
mouse_down = False
mouse_last_x, mouse_last_y = 0, 0
mouse_sensitivity = 0.3

# Window handle
window_id = None

# Animation control
animation_active = False
animation_type = None  # 'hoist', 'lower', 'raise_boom', 'lower_boom'
animation_start_time = 0
animation_duration = 2.0  # seconds
animation_speed = 0
animation_target = 0
animation_start_value = 0
user_input_buffer = ""

# ML Model integration
command_queue = []  # Queue to store commands from ML model or user input
ml_command_callback = None  # Function to get commands from ML model

# Building positions (x, z, width, depth, height)
buildings = [
    (-25, -20, 8, 10, 15, (0.7, 0.7, 0.75)),
    (-25, 15, 10, 8, 20, (0.65, 0.68, 0.72)),
    (20, -25, 12, 8, 18, (0.6, 0.65, 0.7)),
    (25, 10, 8, 12, 22, (0.72, 0.7, 0.68)),
    (-15, -35, 6, 6, 12, (0.68, 0.7, 0.73)),
    (15, 30, 7, 9, 16, (0.63, 0.66, 0.71)),
]


# -------------------- Lighting --------------------
def set_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    
    # Main sun light (warm daylight)
    glLightfv(GL_LIGHT0, GL_POSITION, [20, 40, 20, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 0.95, 0.8, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 0.7, 1])
    
    # Fill light (cooler, from opposite side)
    glLightfv(GL_LIGHT1, GL_POSITION, [-15, 20, -15, 1])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.35, 0.4, 1])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.1, 0.1, 0.1, 1])
    
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.25, 1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20)


# -------------------- Drawing primitives --------------------
def draw_cube(size=1.0):
    s = size / 2.0
    glBegin(GL_QUADS)
    # top
    glNormal3f(0, 1, 0)
    glVertex3f(-s, s, -s); glVertex3f(s, s, -s); glVertex3f(s, s, s); glVertex3f(-s, s, s)
    # bottom
    glNormal3f(0, -1, 0)
    glVertex3f(-s, -s, -s); glVertex3f(-s, -s, s); glVertex3f(s, -s, s); glVertex3f(s, -s, -s)
    # front
    glNormal3f(0, 0, 1)
    glVertex3f(-s, -s, s); glVertex3f(-s, s, s); glVertex3f(s, s, s); glVertex3f(s, -s, s)
    # back
    glNormal3f(0, 0, -1)
    glVertex3f(-s, -s, -s); glVertex3f(s, -s, -s); glVertex3f(s, s, -s); glVertex3f(-s, s, -s)
    # left
    glNormal3f(-1, 0, 0)
    glVertex3f(-s, -s, -s); glVertex3f(-s, s, -s); glVertex3f(-s, s, s); glVertex3f(-s, -s, s)
    # right
    glNormal3f(1, 0, 0)
    glVertex3f(s, -s, -s); glVertex3f(s, -s, s); glVertex3f(s, s, s); glVertex3f(s, s, -s)
    glEnd()


def draw_cylinder(radius, height, slices=16):
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluCylinder(quad, radius, radius, height, slices, 1)
    gluDeleteQuadric(quad)


# -------------------- Scene objects --------------------
def draw_crane_base():
    """Draw a realistic crane base with concrete foundation and support structure"""
    # Concrete foundation
    glPushMatrix()
    glColor3f(0.5, 0.5, 0.52)  # Concrete gray
    glTranslatef(0, 0.3, 0)
    glScalef(4.5, 0.6, 4.5)
    draw_cube(1.0)
    glPopMatrix()
    
    # Steel base platform
    glPushMatrix()
    glColor3f(0.35, 0.35, 0.37)  # Dark steel
    glTranslatef(0, 0.9, 0)
    glScalef(3.5, 0.4, 3.5)
    draw_cube(1.0)
    glPopMatrix()
    
    # Four corner support pillars
    pillar_color = (1.0, 0.65, 0.0)  # Bright safety orange
    for x in [-1.2, 1.2]:
        for z in [-1.2, 1.2]:
            glPushMatrix()
            glColor3f(*pillar_color)
            glTranslatef(x, 1.5, z)
            glScalef(0.3, 2.2, 0.3)
            draw_cube(1.0)
            glPopMatrix()
    
    # Cross bracing
    glColor3f(0.9, 0.55, 0.0)
    glLineWidth(3.0)
    glDisable(GL_LIGHTING)
    glBegin(GL_LINES)
    for x in [-1.2, 1.2]:
        glVertex3f(x, 0.6, -1.2)
        glVertex3f(-x, 0.6, 1.2)
    glEnd()
    glEnable(GL_LIGHTING)
    glLineWidth(1.0)


def draw_crane():
    """Draw the main crane structure"""
    global boom_angle, boom_length, mast_height, hook_length
    
    # Main mast (tower)
    glPushMatrix()
    glColor3f(1.0, 0.65, 0.0)  # Bright safety orange
    glTranslatef(0, mast_height / 2.0 + 2.6, 0)
    glScalef(0.5, mast_height, 0.5)
    draw_cube(1.0)
    glPopMatrix()
    
    # Mast details (side panels)
    for angle in [0, 90, 180, 270]:
        glPushMatrix()
        glRotatef(angle, 0, 1, 0)
        glTranslatef(0.3, mast_height / 2.0 + 2.6, 0)
        glColor3f(0.9, 0.55, 0.0)
        glScalef(0.05, mast_height - 0.5, 0.3)
        draw_cube(1.0)
        glPopMatrix()
    
    # Operator cabin
    glPushMatrix()
    glTranslatef(0, mast_height + 2.6, 0.8)
    glColor3f(0.2, 0.25, 0.3)
    glScalef(1.0, 1.2, 1.0)
    draw_cube(1.0)
    glPopMatrix()
    
    # Cabin windows (dark blue)
    glPushMatrix()
    glTranslatef(0, mast_height + 2.8, 1.32)
    glColor3f(0.1, 0.15, 0.25)
    glScalef(0.7, 0.6, 0.02)
    draw_cube(1.0)
    glPopMatrix()
    
    # Platform at top
    glPushMatrix()
    glTranslatef(0, mast_height + 2.6, 0)
    glColor3f(0.3, 0.3, 0.32)
    glScalef(1.5, 0.2, 1.5)
    draw_cube(1.0)
    glPopMatrix()

    # Boom assembly
    glPushMatrix()
    glTranslatef(0, mast_height + 2.6, 0)
    glRotatef(-boom_angle, 0, 0, 1)
    
    # Main boom arm
    glColor3f(1.0, 0.85, 0.0)  # Bright yellow - very visible
    glPushMatrix()
    glTranslatef(boom_length / 2.0, 0, 0)
    glScalef(boom_length, 0.35, 0.35)
    draw_cube(1.0)
    glPopMatrix()
    
    # Boom truss structure
    glColor3f(0.95, 0.75, 0.0)
    num_segments = 8
    for i in range(num_segments):
        x = (i / float(num_segments)) * boom_length
        glPushMatrix()
        glTranslatef(x, 0.25, 0)
        glScalef(0.15, 0.5, 0.15)
        draw_cube(1.0)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(x, -0.25, 0)
        glScalef(0.15, 0.5, 0.15)
        draw_cube(1.0)
        glPopMatrix()
    
    # Counter weight at boom base
    glPushMatrix()
    glTranslatef(-1.5, 0, 0)
    glColor3f(0.4, 0.4, 0.42)
    glScalef(1.2, 0.8, 0.8)
    draw_cube(1.0)
    glPopMatrix()
    
    # Hook and cable
    glPushMatrix()
    glTranslatef(boom_length, 0, 0)
    
    # Cable
    glDisable(GL_LIGHTING)
    glColor3f(0.15, 0.15, 0.15)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, -hook_length, 0)
    glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)
    
    # Hook assembly
    glTranslatef(0, -hook_length, 0)
    glColor3f(1.0, 0.1, 0.0)  # Bright safety red - highly visible
    
    # Hook body
    glPushMatrix()
    glScalef(0.5, 0.6, 0.3)
    draw_cube(1.0)
    glPopMatrix()
    
    # Hook point
    glPushMatrix()
    glTranslatef(0, -0.5, 0)
    glColor3f(0.85, 0.05, 0.0)
    glScalef(0.3, 0.4, 0.2)
    draw_cube(1.0)
    glPopMatrix()
    
    glPopMatrix()  # End hook
    glPopMatrix()  # End boom


def draw_building(x, z, width, depth, height, color):
    """Draw a stylized building"""
    glPushMatrix()
    glTranslatef(x, height / 2.0, z)
    glColor3f(*color)
    glScalef(width, height, depth)
    draw_cube(1.0)
    glPopMatrix()
    
    # Windows pattern
    glDisable(GL_LIGHTING)
    window_color = (0.3, 0.4, 0.5) if random.random() > 0.3 else (0.9, 0.85, 0.5)
    rows = int(height / 2.5)
    cols_w = int(width / 2.0)
    cols_d = int(depth / 2.0)
    
    # Front and back faces
    for face, z_offset in [(z + depth/2.0 + 0.01, 1), (z - depth/2.0 - 0.01, -1)]:
        for row in range(1, rows):
            for col in range(cols_w):
                wx = x - width/2.0 + (col + 0.5) * (width / cols_w)
                wy = row * (height / rows)
                glColor3f(*window_color)
                glBegin(GL_QUADS)
                glVertex3f(wx - 0.3, wy - 0.4, face)
                glVertex3f(wx + 0.3, wy - 0.4, face)
                glVertex3f(wx + 0.3, wy + 0.4, face)
                glVertex3f(wx - 0.3, wy + 0.4, face)
                glEnd()
    
    # Side faces
    for face, x_offset in [(x + width/2.0 + 0.01, 1), (x - width/2.0 - 0.01, -1)]:
        for row in range(1, rows):
            for col in range(cols_d):
                wz = z - depth/2.0 + (col + 0.5) * (depth / cols_d)
                wy = row * (height / rows)
                glColor3f(*window_color)
                glBegin(GL_QUADS)
                glVertex3f(face, wy - 0.4, wz - 0.3)
                glVertex3f(face, wy - 0.4, wz + 0.3)
                glVertex3f(face, wy + 0.4, wz + 0.3)
                glVertex3f(face, wy + 0.4, wz - 0.3)
                glEnd()
    
    glEnable(GL_LIGHTING)


def draw_ground():
    """Draw construction site ground"""
    # Main ground (dirt/concrete)
    glDisable(GL_LIGHTING)
    glColor3f(0.55, 0.52, 0.48)
    glBegin(GL_QUADS)
    glVertex3f(-60, 0, -60)
    glVertex3f(60, 0, -60)
    glVertex3f(60, 0, 60)
    glVertex3f(-60, 0, 60)
    glEnd()
    
    # Grid pattern on ground
    glColor3f(0.48, 0.45, 0.42)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for i in range(-60, 61, 5):
        glVertex3f(i, 0.01, -60)
        glVertex3f(i, 0.01, 60)
        glVertex3f(-60, 0.01, i)
        glVertex3f(60, 0.01, i)
    glEnd()
    
    glEnable(GL_LIGHTING)


def draw_sky():
    """Draw gradient sky"""
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    # Sky gradient
    glBegin(GL_QUADS)
    glColor3f(0.4, 0.6, 0.9)  # Top (blue)
    glVertex3f(-1, 1, -0.5)
    glVertex3f(1, 1, -0.5)
    glColor3f(0.7, 0.8, 0.95)  # Bottom (lighter)
    glVertex3f(1, -0.3, -0.5)
    glVertex3f(-1, -0.3, -0.5)
    glEnd()
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


# -------------------- Display --------------------
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    draw_sky()
    
    glEnable(GL_DEPTH_TEST)
    glLoadIdentity()

    # Camera
    az, el = math.radians(cam_azimuth), math.radians(cam_elevation)
    cam_x = cam_distance * math.cos(el) * math.sin(az)
    cam_y = cam_distance * math.sin(el)
    cam_z = cam_distance * math.cos(el) * math.cos(az)
    gluLookAt(cam_x, cam_y + 5, cam_z, 0, 6, 0, 0, 1, 0)

    set_lighting()
    
    draw_ground()
    
    # Draw buildings
    for bld in buildings:
        draw_building(*bld)
    
    # Draw crane
    draw_crane_base()
    draw_crane()

    # Draw HUD
    draw_hud()
    
    glutSwapBuffers()


def draw_hud():
    """Draw on-screen controls info"""
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, window_width, 0, window_height)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glColor3f(1, 1, 1)
    text = [
        "CRANE CONTROLS: Arrow Keys (Boom/Hook) or Type Commands",
        "CAMERA: Drag Mouse (Rotate) | Mouse Wheel (Zoom) | A/D/W/S (Alt Control)",
        f"Boom Angle: {boom_angle:.1f}deg  Hook Length: {hook_length:.1f}m",
        "Commands: 'hoist', 'lower', 'raise boom', 'lower boom' + ENTER",
        "ESC or Q: Exit"
    ]
    
    y = window_height - 25
    for line in text:
        glRasterPos2f(10, y)
        for char in line:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
        y -= 20
    
    # Display current input
    if user_input_buffer:
        glColor3f(0.2, 0.8, 0.2)
        glRasterPos2f(10, 80)
        input_text = f"Command: {user_input_buffer}_"
        for char in input_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
    
    # Display active animation
    if animation_active:
        glColor3f(1, 1, 0)
        glRasterPos2f(10, 50)
        anim_text = f"EXECUTING: {animation_type.replace('_', ' ').upper()}"
        for char in anim_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
        
        # Progress bar
        elapsed = time.time() - animation_start_time
        progress = min(1.0, elapsed / animation_duration)
        bar_width = 300
        bar_height = 20
        
        # Background
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex2f(10, 25)
        glVertex2f(10 + bar_width, 25)
        glVertex2f(10 + bar_width, 25 + bar_height)
        glVertex2f(10, 25 + bar_height)
        glEnd()
        
        # Progress
        glColor3f(0.2, 0.8, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(10, 25)
        glVertex2f(10 + bar_width * progress, 25)
        glVertex2f(10 + bar_width * progress, 25 + bar_height)
        glVertex2f(10, 25 + bar_height)
        glEnd()
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


# -------------------- Event Handlers --------------------
def reshape(w, h):
    global window_width, window_height
    window_width, window_height = w, h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(50.0, w / float(h if h != 0 else 1), 0.1, 300.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def start_animation(anim_type):
    """Start an automated crane operation"""
    global animation_active, animation_type, animation_start_time, animation_speed
    global animation_target, animation_start_value, boom_angle, hook_length
    
    animation_active = True
    animation_type = anim_type
    animation_start_time = time.time()
    
    # Set target values for noticeable change
    if anim_type == 'hoist':
        animation_start_value = hook_length
        animation_target = max(min_hook, hook_length - 4.0)  # Raise 4 meters
    elif anim_type == 'lower':
        animation_start_value = hook_length
        animation_target = min(max_hook, hook_length + 4.0)  # Lower 4 meters
    elif anim_type == 'raise_boom':
        animation_start_value = boom_angle
        animation_target = min(max_boom_angle, boom_angle + 20.0)  # Raise 20 degrees
    elif anim_type == 'lower_boom':
        animation_start_value = boom_angle
        animation_target = max(min_boom_angle, boom_angle - 20.0)  # Lower 20 degrees


def execute_command(command_str):
    """Execute a crane command from user input or ML model"""
    cmd = command_str.lower().strip()
    if cmd == 'hoist':
        start_animation('hoist')
        return True
    elif cmd == 'lower':
        start_animation('lower')
        return True
    elif cmd in ['raise boom', 'raiseboom', 'raise_boom']:
        start_animation('raise_boom')
        return True
    elif cmd in ['lower boom', 'lowerboom', 'lower_boom']:
        start_animation('lower_boom')
        return True
    return False


def set_ml_command_callback(callback_func):
    """
    Set a callback function that returns ML model predictions.
    The callback should return a command string or None.
    
    Example usage:
        def my_ml_model():
            # Your ML model code here
            prediction = model.predict(current_state)
            return prediction  # Should return 'hoist', 'lower', 'raise_boom', or 'lower_boom'
        
        set_ml_command_callback(my_ml_model)
    """
    global ml_command_callback
    ml_command_callback = callback_func


def add_command_to_queue(command):
    """Add a command to the execution queue (for ML model or external input)"""
    global command_queue
    command_queue.append(command)


def process_command_queue():
    """Process commands from the queue if no animation is active"""
    global command_queue, animation_active
    
    if not animation_active and command_queue:
        next_command = command_queue.pop(0)
        execute_command(next_command)


def update_animation():
    """Update crane position during animation"""
    global animation_active, boom_angle, hook_length
    
    if not animation_active:
        return
    
    elapsed = time.time() - animation_start_time
    
    if elapsed >= animation_duration:
        # Snap to final target value
        if animation_type in ['hoist', 'lower']:
            hook_length = animation_target
        elif animation_type in ['raise_boom', 'lower_boom']:
            boom_angle = animation_target
        animation_active = False
        return
    
    # Calculate smooth interpolation (easing)
    progress = elapsed / animation_duration
    # Apply ease-in-out for smooth motion
    if progress < 0.5:
        eased_progress = 2 * progress * progress
    else:
        eased_progress = 1 - pow(-2 * progress + 2, 2) / 2
    
    # Interpolate between start and target
    if animation_type == 'hoist' or animation_type == 'lower':
        hook_length = animation_start_value + (animation_target - animation_start_value) * eased_progress
    elif animation_type == 'raise_boom' or animation_type == 'lower_boom':
        boom_angle = animation_start_value + (animation_target - animation_start_value) * eased_progress


def get_crane_state():
    """
    Get current crane state for ML model input.
    Returns a dictionary with current crane parameters.
    """
    return {
        'boom_angle': boom_angle,
        'hook_length': hook_length,
        'animation_active': animation_active,
        'animation_type': animation_type if animation_active else None,
        'cam_azimuth': cam_azimuth,
        'cam_elevation': cam_elevation,
        'cam_distance': cam_distance
    }


def keyboard(key, x, y):
    global cam_azimuth, cam_elevation, cam_distance, window_id, user_input_buffer
    k = key.decode("utf-8")
    
    if k == '\x1b' or k == 'q':  # ESC or Q
        if window_id is not None:
            glutDestroyWindow(window_id)
        import os
        os._exit(0)
    elif k == '\r':  # ENTER key
        # Process command from user input
        if user_input_buffer:
            success = execute_command(user_input_buffer)
            if not success:
                print(f"Unknown command: {user_input_buffer}")
            user_input_buffer = ""
    elif k == '\b':  # BACKSPACE
        if user_input_buffer:
            user_input_buffer = user_input_buffer[:-1]
    elif k.isalnum() or k == ' ':  # Allow letters, numbers, and spaces
        user_input_buffer += k
    elif k == 'a':
        cam_azimuth -= cam_step
    elif k == 'd':
        cam_azimuth += cam_step
    elif k == 'w':
        cam_elevation = min(89.0, cam_elevation + cam_step)
    elif k == 's':
        cam_elevation = max(-10.0, cam_elevation - cam_step)
    elif k == 'z':
        cam_distance = max(8.0, cam_distance - zoom_step)
    elif k == 'x':
        cam_distance = min(80.0, cam_distance + zoom_step)
    glutPostRedisplay()


def special_keys(key, x, y):
    global boom_angle, hook_length
    if key == GLUT_KEY_LEFT:
        boom_angle = max(min_boom_angle, boom_angle - angle_step)
    elif key == GLUT_KEY_RIGHT:
        boom_angle = min(max_boom_angle, boom_angle + angle_step)
    elif key == GLUT_KEY_UP:
        hook_length = max(min_hook, hook_length - hook_step)
    elif key == GLUT_KEY_DOWN:
        hook_length = min(max_hook, hook_length + hook_step)
    glutPostRedisplay()


def mouse_button(button, state, x, y):
    global mouse_down, mouse_last_x, mouse_last_y, cam_distance
    
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            mouse_down = True
            mouse_last_x = x
            mouse_last_y = y
        else:
            mouse_down = False
    
    # Mouse wheel for zoom
    elif button == 3:  # Scroll up
        cam_distance = max(8.0, cam_distance - zoom_step * 2)
        glutPostRedisplay()
    elif button == 4:  # Scroll down
        cam_distance = min(80.0, cam_distance + zoom_step * 2)
        glutPostRedisplay()


def mouse_motion(x, y):
    global cam_azimuth, cam_elevation, mouse_last_x, mouse_last_y
    
    if mouse_down:
        dx = x - mouse_last_x
        dy = y - mouse_last_y
        
        cam_azimuth += dx * mouse_sensitivity
        cam_elevation -= dy * mouse_sensitivity
        
        # Clamp elevation
        cam_elevation = max(-10.0, min(89.0, cam_elevation))
        
        mouse_last_x = x
        mouse_last_y = y
        
        glutPostRedisplay()


def idle():
    global ml_command_callback
    
    # Check if ML model has a new command
    if ml_command_callback is not None and not animation_active:
        try:
            ml_command = ml_command_callback()
            if ml_command:
                add_command_to_queue(ml_command)
        except Exception as e:
            print(f"ML callback error: {e}")
    
    # Process queued commands
    process_command_queue()
    
    # Update animation
    update_animation()
    glutPostRedisplay()


def main():
    global window_id
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    window_id = glutCreateWindow(b"Construction Crane Simulator")
    glClearColor(0.7, 0.8, 0.95, 1.0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)
    glutMouseFunc(mouse_button)
    glutMotionFunc(mouse_motion)
    glutIdleFunc(idle)
    glutMainLoop()


if __name__ == "__main__":
    # Example: Integrate with ML model
    # Uncomment and modify this section to connect your ML model:
    
    # def my_ml_model():
    #     """Your ML model prediction function"""
    #     state = get_crane_state()
    #     # Process state with your model
    #     # prediction = your_model.predict(state)
    #     # return prediction  # Should return: 'hoist', 'lower', 'raise_boom', or 'lower_boom'
    #     return None
    # 
    # set_ml_command_callback(my_ml_model)
    
    # Alternative: Send commands directly from external code
    # import threading
    # def send_commands():
    #     time.sleep(3)
    #     add_command_to_queue('raise_boom')
    #     time.sleep(3)
    #     add_command_to_queue('hoist')
    # threading.Thread(target=send_commands, daemon=True).start()
    
    main()