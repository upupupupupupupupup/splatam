import numpy as np
import open3d as o3d

colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
translates = [-1, -0.5, 0, 0.5, 1]


class Person:
    def __init__(self, rd, color_id):
        self.head = o3d.geometry.TriangleMesh.create_sphere(radius=2 * rd)
        self.head.compute_vertex_normals()
        self.head.paint_uniform_color(colors[color_id % 3])
        self.body = mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=rd, height=4 * rd
        )
        self.body.compute_vertex_normals()
        self.body.paint_uniform_color(colors[(color_id + 1) % 3])

    def add(self, vis):
        vis.add_geometry(self.head)
        vis.add_geometry(self.body)

    def delete(self, vis):
        vis.remove_geometry(self.head)
        vis.remove_geometry(self.body)

    def translate(self, center):
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = center
        self.head.transform(translation_matrix)
        translation_matrix[:3, 3] = center + 3
        self.body.transform(np.eye(4))


vis = o3d.visualization.Visualizer()
vis.create_window()

# ctr = vis.get_view_control()
# ctr.set_lookat(np.array([0.0, 0.0, 55.0]))
# ctr.set_up((0, -1, 0))  # set the positive direction of the x-axis as the up direction
# ctr.set_front((-1, 0, 0))  # set the positive direction of the x-axis toward you

color_id = 0
translate_id = 0
persons = []
center = np.array([1.0, 1.0, 1.0])

while True:

    color_id += 1
    translate_id += 1
    # center += 0.01

    if len(persons):
        for person in persons:
            person.delete(vis)
        del person

    for i in range(2):

        person = Person(rd=0.1 + 0.1 * i, color_id=color_id + i)

        person.translate(translates[translate_id % 5] + i)

        person.add(vis)

        persons.append(person)

    if not vis.poll_events():
        break
    vis.update_renderer()

vis.destroy_window()
