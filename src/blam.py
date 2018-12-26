#
# blam - Blender Camera Calibration Tools
# Copyright (C) 2012-2014  Per Gantelius
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/
#

import bpy
from mathutils import Vector, Matrix, Euler
from math import tan, sqrt, radians
import numpy as np

bl_info = {
    'name': 'BLAM - The Blender camera calibration toolkit',
    'author': 'Per Gantelius',
    'version': (0, 0, 6),
    'blender': (2, 78, 0),
    'location': 'Move Clip Editor > Tools > Static Camera Calibration and 3D View > Tool Shelf > Photo Modeling Tools',
    'description': 'Reconstruct 3D geometry and estimate camera orientation and focal length based on photographs',
    'tracker_url': 'https://github.com/stuffmatic/blam/issues',
    'wiki_url': 'https://github.com/stuffmatic/blam/wiki',
    'support': 'COMMUNITY',
    'category': '3D View'
}


#
# CAMERA CALIBRATION STUFF
#

class BLAM_PT_photo_modeling_tools(bpy.types.Panel):
    bl_label = "Photo Modeling Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"

    def draw(self, context):
        layout = self.layout
        props = context.scene.blam

        row = layout.row()
        box = row.box()
        box.operator("blam.reconstruct_mesh_with_rects", icon='MESH_CUBE')
        box.prop(props, "separate_faces")

        row = layout.row()
        box = row.box()
        box.operator("blam.project_bg_onto_mesh", icon='MOD_UVPROJECT')
        box.prop(props, "projection_method")

        # self.layout.operator("blam.make_edge_x")
        layout.operator("blam.set_pivot_to_camera", icon='CURSOR')


class BLAM_OT_set_pivot_to_camera(bpy.types.Operator):
    bl_idname = "blam.set_pivot_to_camera"
    bl_label = "Set Pivot to Camera Origin"
    bl_description = "Set the pivot to the origin of the active camera so that " \
                     "the screen position of each vertex is invariant to scaling in 3D space"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # snap the cursor to the camera
        context.scene.cursor_location = context.scene.camera.location

        # set the cursor to be the pivot
        space = context.area.spaces.active
        print(space.pivot_point)
        space.pivot_point = 'CURSOR'

        return {'FINISHED'}


class BLAM_OT_project_bg_onto_mesh(bpy.types.Operator):
    bl_idname = "blam.project_bg_onto_mesh"
    bl_label = "Project Background Image onto Mesh"
    bl_description = "Projects the current 3D view background image onto a mesh (the active object) " \
                     "from the active camera"
    bl_options = {'REGISTER', 'UNDO'}

    projectorName = 'tex_projector'
    materialName = 'cam_map_material'

    def meshVerticesToNDC(self, context, cam, mesh):

        # compute a projection matrix transforming
        # points in camera space to points in NDC
        near = cam.data.clip_start
        far = cam.data.clip_end
        sx = 2 * cam.data.shift_x
        sy = 2 * cam.data.shift_y
        rs = context.scene.render
        rx = rs.resolution_x
        ry = rs.resolution_y
        sf = cam.data.sensor_fit
        if sf == 'AUTO' and rx < ry or sf == 'VERTICAL':
            fov = cam.data.angle
            aspect = rx / ry
            h = tan(0.5 * fov)
            w = aspect * h
            sx /= aspect
        else:
            fov = cam.data.angle
            aspect = ry / rx
            w = tan(0.5 * fov)
            h = aspect * w
            sy /= aspect

        pm = Matrix()
        pm[0][0] = 1 / w
        pm[1][1] = 1 / h
        pm[2][2] = (near + far) / (near - far)
        pm[2][3] = 2 * near * far / (near - far)
        pm[3][2] = -1.0
        pm[3][3] = 0.0

        returnVerts = []

        for v in mesh.data.vertices:
            # the vert in local coordinates
            vec = v.co.to_4d()
            # the vert in world coordinates
            vec = mesh.matrix_world * vec
            # the vert in clip coordinates
            vec = pm * cam.matrix_world.inverted() * vec
            # the vert in normalized device coordinates
            vec = vec.to_3d() / vec.w
            returnVerts.append(vec - Vector((sx, sy, 0)))

        return returnVerts

    def addUVsProjectedFromView(self, context, cam, mesh):

        # get the mesh vertices in normalized device coordinates
        # as seen through the active camera
        ndcVerts = self.meshVerticesToNDC(context, cam, mesh)

        # create a uv layer
        bpy.ops.object.mode_set(mode='EDIT')
        # projecting from view here, but the current view might not
        # be the camera, so the uvs are computed manually a couple
        # of lines down
        bpy.ops.uv.project_from_view(scale_to_bounds=True)
        bpy.ops.object.mode_set(mode='OBJECT')

        # set uvs to match the vertex x and y components in NDC
        loops = mesh.data.loops
        uvLoops = mesh.data.uv_layers[0].data
        for loop, uvLoop in zip(loops, uvLoops):
            vIdx = loop.vertex_index
            print("loop", loop, "vertex", loop.vertex_index, "uvLoop", uvLoop)
            uvLoop.uv = 0.5 * (ndcVerts[vIdx].to_2d() + Vector((1.0, 1.0)))

    def performSimpleProjection(self, context, camera, mesh, img):
        if len(mesh.material_slots) == 0:
            mat = bpy.data.materials.new(self.materialName)
            mesh.data.materials.append(mat)
        else:
            mat = mesh.material_slots[0].material
        mat.use_shadeless = True
        mat.use_face_texture = True

        self.addUVsProjectedFromView(context, camera, mesh)

        for f in mesh.data.uv_textures[0].data:
            f.image = img

    def performHighQualityProjection(self, context, camera, mesh, img):
        if len(mesh.material_slots) == 0:
            mat = bpy.data.materials.new(self.materialName)
            mesh.data.materials.append(mat)
        else:
            mat = mesh.material_slots[0].material
        mat.use_shadeless = True
        mat.use_face_texture = True

        # the texture sampling is not perspective correct
        # when directly using sticky UVs or UVs projected from the view
        # this is a pretty messy workaround that gives better looking results
        self.addUVsProjectedFromView(context, camera, mesh)

        # then create an empty object that will serve as a texture projector
        # if the mesh has a child with the name of a texture projector,
        # reuse it
        reusedProjector = None
        for ch in mesh.children:
            if self.projectorName in ch.name:
                reusedProjector = ch
                break

        if reusedProjector:
            projector = reusedProjector
        else:
            bpy.ops.object.camera_add()
            projector = context.active_object

        context.scene.objects.active = projector
        projector.name = mesh.name + '_' + self.projectorName
        projector.matrix_world = camera.matrix_world
        projector.select = False
        projector.scale = Vector((0.1, 0.1, 0.1))
        projector.data.lens = camera.data.lens
        projector.data.shift_x = camera.data.shift_x
        projector.data.shift_y = camera.data.shift_y
        projector.data.sensor_width = camera.data.sensor_width
        projector.data.sensor_height = camera.data.sensor_height
        projector.data.sensor_fit = camera.data.sensor_fit

        # parent the projector to the mesh for convenience
        for obj in context.scene.objects:
            obj.select = False

        projector.select = True
        context.scene.objects.active = mesh
        # bpy.ops.object.parent_set()

        # lock the projector to the mesh
        # context.scene.objects.active = projector
        # bpy.ops.object.constraint_add(type='COPY_LOCATION')
        # projector.constraints[-1].target = mesh

        # create a simple subdivision modifier on the mesh object.
        # this subdivision is what alleviates the texture sampling
        # artefacts.
        context.scene.objects.active = mesh
        levels = 3
        bpy.ops.object.modifier_add()

        modifier = mesh.modifiers[-1]
        modifier.subdivision_type = 'SIMPLE'
        modifier.levels = levels
        modifier.render_levels = levels

        # then create a uv project modifier that will project the
        # image onto the subdivided mesh using our projector object.
        bpy.ops.object.modifier_add(type='UV_PROJECT')
        modifier = mesh.modifiers[-1]
        modifier.aspect_x = context.scene.render.resolution_x / context.scene.render.resolution_y
        modifier.aspect_y = context.scene.render.resolution_y / context.scene.render.resolution_x
        modifier.image = img
        modifier.use_image_override = True
        modifier.projectors[0].object = projector
        modifier.uv_layer = mesh.data.uv_textures[0].name

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')

    def prepareMesh(self, mesh):
        # remove all uv layers
        while len(mesh.data.uv_textures) > 0:
            bpy.ops.mesh.uv_texture_remove()

        # remove all modifiers
        for m in mesh.modifiers:
            bpy.ops.object.modifier_remove(modifier=m.name)

    def execute(self, context):
        props = context.scene.blam

        #
        # Get the active object and make sure it is a mesh
        #
        mesh = context.active_object

        if mesh is None:
            self.report({'ERROR'}, "There is no active object")
            return {'CANCELLED'}
        elif 'Mesh' not in str(type(mesh.data)):
            self.report({'ERROR'}, "The active object is not a mesh")
            return {'CANCELLED'}

        #
        # Get the current camera
        #
        camera = context.scene.camera
        if not camera:
            self.report({'ERROR'}, "No active camera.")
            return {'CANCELLED'}

        activeSpace = context.space_data

        if len(activeSpace.background_images) == 0:
            self.report({'ERROR'}, "No backround images of clips found.")
            return {'CANCELLED'}

        # check what kind of background we're dealing with
        bg = activeSpace.background_images[0]
        if bg.image is not None:
            img = bg.image
        elif bg.clip is not None:
            path = bg.clip.filepath
            # create an image texture from the (first) background clip
            try:
                img = bpy.data.images.load(path)
            except RuntimeError:
                self.report({'ERROR'}, "Cannot load image %s" % path)
                return {'CANCELLED'}
        else:
            # shouldnt end up here
            self.report({'ERROR'}, "Both background clip and image are None")
            return {'CANCELLED'}

        # if we made it here, we have a camera, a mesh and an image.
        self.prepareMesh(mesh)
        method = props.projection_method
        if method == 'HQ':
            self.performHighQualityProjection(context, camera, mesh, img)
        elif method == 'SIMPLE':
            self.performSimpleProjection(context, camera, mesh, img)
        else:
            self.report({'ERROR'}, "Unknown projection method")
            return {'CANCELLED'}

        activeSpace.viewport_shade = 'TEXTURED'

        return {'FINISHED'}


class BLAM_OT_reconstruct_mesh_with_rects(bpy.types.Operator):
    bl_idname = "blam.reconstruct_mesh_with_rects"
    bl_label = "Reconstruct 3D Geometry"
    bl_description = "Reconstructs a 3D mesh with rectangular faces " \
                     "based on a mesh with faces lining up with the corresponding faces in the image. " \
                     "Relies on the active camera being properly calibrated"
    bl_options = {'REGISTER', 'UNDO'}

    def evalEq17(self, origin, p1, p2):
        a = origin - p1
        b = origin - p2
        return a.dot(b)

    def evalEq27(self, l):
        return self.C4 * l**4 + self.C3 * l**3 + self.C2 * l**2 + self.C1 * l + self.C0

    def evalEq28(self, l):
        return self.B4 * l**4 + self.B3 * l**3 + self.B2 * l**2 + self.B1 * l + self.B0

    def evalEq29(self, l):
        return self.D3 * l**3 + self.D2 * l**2 + self.D1 * l + self.D0

    def worldToCameraSpace(self, verts):

        ret = []
        for v in verts:
            # the vert in local coordinates
            vec = v.co.to_4d()
            # the vert in world coordinates
            vec = self.mesh.matrix_world * vec
            # the vert in camera coordinates
            vec = self.camera.matrix_world.inverted() * vec

            ret.append(vec.to_3d())
        return ret

    def computeCi(self, Qab, Qac, Qad, Qbc, Qbd, Qcd):
        self.C4 = Qad*Qbc*Qbd - Qac*Qbd*Qbd
        self.C3 = (Qab*Qad*Qbd*Qcd - Qad*Qad*Qcd + Qac*Qad*Qbd*Qbd - Qad*Qad*Qbc*Qbd - Qbc*Qbd +
                   2*Qab*Qac*Qbd - Qab*Qad*Qbc)
        self.C2 = (-Qab*Qbd*Qcd - Qab*Qab*Qad*Qcd + 2*Qad*Qcd + Qad*Qbc*Qbd - 3*Qab*Qac*Qad*Qbd +
                   Qab*Qad*Qad*Qbc + Qab*Qbc + Qac*Qad*Qad - Qab*Qab*Qac)
        self.C1 = Qab*Qab*Qcd - Qcd + Qab*Qac*Qbd - Qab*Qad*Qbc + 2*Qab*Qab*Qac*Qad - 2*Qac*Qad
        self.C0 = Qac - Qab*Qab*Qac

    def computeBi(self, Qab, Qac, Qad, Qbc, Qbd, Qcd):
        self.B4 = Qbd - Qbd*Qcd*Qcd
        self.B3 = 2*Qad*Qbd*Qcd*Qcd + Qab*Qcd*Qcd + Qac*Qbd*Qcd - Qad*Qbc*Qcd - 2*Qad*Qbd - Qab
        self.B2 = (-Qbd*Qcd*Qcd - Qab*Qad*Qcd*Qcd - 3*Qac*Qad*Qbd*Qcd + Qad*Qad*Qbc*Qcd +
                   Qbc*Qcd - Qab*Qac*Qcd + Qad*Qad*Qbd + Qac*Qad*Qbc + 2*Qab*Qad)
        self.B1 = (2*Qac*Qbd*Qcd - Qad*Qbc*Qcd + Qab*Qac*Qad*Qcd +
                   Qac*Qac*Qad*Qbd - Qac*Qad*Qad*Qbc - Qac*Qbc - Qab*Qad*Qad)
        self.B0 = Qac*Qad*Qbc - Qac*Qac*Qbd

    def computeQuadDepthInformation(self, qHatA, qHatB, qHatC, qHatD):

        # print()
        # print("computeQuadDepthInformation")

        #
        # compute the coefficients Qij
        #
        Qab = qHatA.dot(qHatB)
        Qac = qHatA.dot(qHatC)
        Qad = qHatA.dot(qHatD)

        # Qba = qHatB.dot(qHatA)
        Qbc = qHatB.dot(qHatC)
        Qbd = qHatB.dot(qHatD)

        # Qca = qHatC.dot(qHatA)
        # Qcb = qHatC.dot(qHatB)
        Qcd = qHatC.dot(qHatD)

        # print("Qab", Qab, "Qac", Qac, "Qad", Qad)
        # print("Qba", Qba, "Qbc", Qbc, "Qbd", Qbd)
        # print("Qca", Qca, "Qcb", Qcb, "Qcd", Qcd)

        #
        # compute the coefficients Ci of equation (27)
        #
        self.computeCi(Qab, Qac, Qad, Qbc, Qbd, Qcd)

        #
        # compute the coefficients Bi of equation (28)
        #
        self.computeBi(Qab, Qac, Qad, Qbc, Qbd, Qcd)

        #
        # compute the cofficients Di of equation (29)
        #
        self.D3 = (self.C4 / self.B4) * self.B3 - self.C3
        self.D2 = (self.C4 / self.B4) * self.B2 - self.C2
        self.D1 = (self.C4 / self.B4) * self.B1 - self.C1
        self.D0 = (self.C4 / self.B4) * self.B0 - self.C0
        # print("Di", self.D3, self.D2, self.D1, self.D0)

        #
        # solve eq 29 for lambdaD, i.e the depth in camera space of vertex D.
        #
        roots = [r.real if r.imag == 0 else complex(r) for r in np.roots([self.D3, self.D2, self.D1, self.D0])]
        # print("Eq 29 Roots", roots)

        # choose one of the three computed roots. Tan, Sullivan and Baker propose
        # choosing a real root that satisfies "(27) or (28)". Since these
        # equations are derived from the orthogonality equations (17) and
        # since we're interested in a quad with edges that are "as
        # orthogonal as possible", in this implementation the positive real
        # root that minimizes the quad orthogonality error is chosen instead.

        chosenRoot = None
        minError = None

        # print()
        # print("Finding root")
        for root in roots:

            # print("Root", root)

            if isinstance(root, complex):
                # complex root. do nothing
                continue
            elif root <= 0:
                # non-positive root. do nothing
                continue

            # compute depth values lambdaA-D based on the current root
            lambdaD = root
            self.lambdaA = 1  # arbitrarily set to 1
            numLambdaA = (Qad * lambdaD - 1.0)
            denLambdaA = (Qbd * lambdaD - Qab)
            self.lambdaB = numLambdaA / denLambdaA
            numLambdaC = (Qad * lambdaD - lambdaD * lambdaD)
            denLambdaC = (Qac - Qcd * lambdaD)
            self.lambdaC = numLambdaC / denLambdaC
            self.lambdaD = lambdaD

            # print("lambdaA", numLambdaA, "/", denLambdaA)
            # print("lambdaC", numLambdaC, "/", denLambdaC)

            # compute vertex positions
            pA = qHatA * self.lambdaA
            pB = qHatB * self.lambdaB
            pC = qHatC * self.lambdaC
            pD = qHatD * self.lambdaD

            # compute the mean orthogonality error for the resulting quad
            meanError, maxError = self.getQuadError(pA, pB, pC, pD)

            if minError is None or meanError < minError:
                minError = meanError
                chosenRoot = root

        if chosenRoot is None:
            self.report({'ERROR'}, "No appropriate root found.")
            return  # TODO cancel properly

        # print("Chosen root", chosenRoot)

        #
        # compute and return the final vertex positions from equation (16)
        #
        lambdaD = chosenRoot
        self.lambdaA = 1  # arbitrarily set to 1
        self.lambdaB = (Qad * lambdaD - 1.0) / (Qbd * lambdaD - Qab)
        self.lambdaC = (Qad * lambdaD - lambdaD * lambdaD) / (Qac - Qcd * lambdaD)
        self.lambdaD = lambdaD

        pA = qHatA * self.lambdaA
        pB = qHatB * self.lambdaB
        pC = qHatC * self.lambdaC
        pD = qHatD * self.lambdaD

        meanError, maxError = self.getQuadError(pA, pB, pC, pD)
        # self.report({'INFO'}, "Error: " + str(meanError) + " (" + str(maxError) + ")")

        return [pA, pB, pC, pD]

    def getQuadError(self, pA, pB, pC, pD):
        orthABD = self.evalEq17(pA, pB, pD)
        orthABC = self.evalEq17(pB, pA, pC)
        orthBCD = self.evalEq17(pC, pB, pD)
        orthACD = self.evalEq17(pD, pA, pC)

        absErrors = [abs(orthABD), abs(orthABC), abs(orthBCD), abs(orthACD)]
        maxError = max(absErrors)
        meanError = 0.25 * sum(absErrors)
        # print("absErrors", absErrors, "meanError", meanError)

        return meanError, maxError

    def createMesh(self, context, inputMesh, computedCoordsByFace, quads, mergeVertices):

        # Mesh creation is done in two steps:
        # 1. adjust the computed depth values so that the
        #    quad vertices line up as well as possible
        # 2. optionally merge each set of computed quad vertices
        #    that correspond to a single vertex in the input mesh

        #
        # Step 1
        # least squares minimize the depth difference
        # at each vertex of each shared edge by
        # computing depth factors for each quad.
        #

        quadFacePairsBySharedEdge = {}

        def indexOfFace(fcs, face):
            for i, f in enumerate(fcs):
                if f == face:
                    return i
            assert(False)  # could not find the face. should not end up here...

        # loop over all edges...
        unsharedEdgeCount = 0  # the number of edges beloning to less than two faces
        quadFaces = []
        for e in inputMesh.data.edges:
            ev = e.vertices

            # gather all faces containing the current edge
            facesContainingEdge = []

            for f in inputMesh.data.polygons:
                matchFound = False
                fv = f.vertices
                if len(fv) != 4:
                    # ignore non-quad faces
                    continue

                if f not in quadFaces:
                    quadFaces.append(f)

                # if the intersection of the face vertices and the
                # print("fv", fv, "ev", ev, "len(set(fv) & set(ev))", len(set(fv) & set(ev)))
                if len(set(fv) & set(ev)) == 2:
                    matchFound = True

                if matchFound:
                    assert(f not in facesContainingEdge)
                    facesContainingEdge.append(f)

            # sanity check. an edge can be shared by at most two faces.
            assert(0 <= len(facesContainingEdge) <= 2)

            edgeIsShared = (len(facesContainingEdge) == 2)

            if edgeIsShared:
                quadFacePairsBySharedEdge[e] = facesContainingEdge
            else:
                unsharedEdgeCount += 1
        numSharedEdges = len(quadFacePairsBySharedEdge.keys())
        numQuadFaces = len(quadFaces)
        # sanity check: the shared and unshared edges are disjoint and should add up to the total number of edges
        assert(unsharedEdgeCount + numSharedEdges == len(inputMesh.data.edges))
        # print("num shared edges", numSharedEdges)
        # print(quadFacePairsBySharedEdge)
        # assert(False)

        # each shared edge gives rise to one equation per vertex,
        # so the number of rows in the matrix is 2n, where n is
        # the number of shared edges. the number of columns is m-1
        # where m is the number of faces (the depth factor for the first
        # face is set to 1)
        # k1 = 1
        # firstFace = inputMesh.data.polygons[0]
        numFaces = len(inputMesh.data.polygons)
        faces = [f for f in inputMesh.data.polygons]
        matrixRows = []
        rhRows = []  # rows of the right hand side vector
        vertsToMergeByOriginalIdx = {}
        for e in quadFacePairsBySharedEdge.keys():
            pair = quadFacePairsBySharedEdge[e]

            # the two original mesh faces sharing the current edge
            f0 = pair[0]
            f1 = pair[1]

            assert(f0 in faces)
            assert(f1 in faces)
            assert(len(f0.vertices) == 4)
            assert(len(f1.vertices) == 4)

            # the two computed quads corresponding to the original mesh faces
            c0 = computedCoordsByFace[f0]
            c1 = computedCoordsByFace[f1]

            # the indices into the output mesh of the two faces sharing the current edge
            f0Idx = quads.index(c0)  # indexOfFace(faces, f0)
            f1Idx = quads.index(c1)  # indexOfFace(faces, f1)

            def getQuadVertWithMeshIdx(quad, idx):
                # print("idx", idx, "quad", quad)
                for i, p in enumerate(quad):
                    if p.w == idx:
                        return p.to_3d(), i
                assert(False)  # shouldnt end up here

            # vij is vertex j of the current edge in face i
            # idxij is the index of vertex j in quad i (0-3)
            v00, idx00 = getQuadVertWithMeshIdx(c0, e.vertices[0])
            v01, idx01 = getQuadVertWithMeshIdx(c0, e.vertices[1])

            v10, idx10 = getQuadVertWithMeshIdx(c1, e.vertices[0])
            v11, idx11 = getQuadVertWithMeshIdx(c1, e.vertices[1])

            # vert 0 depths
            lambda00 = v00.z
            lambda10 = v10.z

            # vert 1 depths
            lambda01 = v01.z
            lambda11 = v11.z
            # print(faces, f0, f1)

            assert(0 <= f0Idx < numFaces)
            assert(0 <= f1Idx < numFaces)

            # vert 0
            vert0MatrixRow = [0] * (numQuadFaces - 1)
            vert0RhRow = [0]

            if f0Idx == 0:
                vert0RhRow[0] = lambda00
                vert0MatrixRow[f1Idx - 1] = lambda10
            elif f1Idx == 0:
                vert0RhRow[0] = lambda10
                vert0MatrixRow[f0Idx - 1] = lambda00
            else:
                vert0MatrixRow[f0Idx - 1] = lambda00
                vert0MatrixRow[f1Idx - 1] = -lambda10

            # vert 1
            vert1MatrixRow = [0] * (numQuadFaces - 1)
            vert1RhRow = [0]

            if f0Idx == 0:
                vert1RhRow[0] = lambda01
                vert1MatrixRow[f1Idx - 1] = lambda11
            elif f1Idx == 0:
                vert1RhRow[0] = lambda11
                vert1MatrixRow[f0Idx - 1] = lambda01
            else:
                vert1MatrixRow[f0Idx - 1] = lambda01
                vert1MatrixRow[f1Idx - 1] = -lambda11

            matrixRows.append(vert0MatrixRow)
            matrixRows.append(vert1MatrixRow)

            rhRows.append(vert0RhRow)
            rhRows.append(vert1RhRow)

            # store index information for vertex merging in the new mesh
            if e.vertices[0] not in vertsToMergeByOriginalIdx.keys():
                vertsToMergeByOriginalIdx[e.vertices[0]] = []
            if e.vertices[1] not in vertsToMergeByOriginalIdx.keys():
                vertsToMergeByOriginalIdx[e.vertices[1]] = []

            l0 = vertsToMergeByOriginalIdx[e.vertices[0]]
            l1 = vertsToMergeByOriginalIdx[e.vertices[1]]

            if idx00 + 4*f0Idx not in l0:
                l0.append(idx00 + 4*f0Idx)
            if idx10 + 4*f1Idx not in l0:
                l0.append(idx10 + 4*f1Idx)

            if idx01 + 4*f0Idx not in l1:
                l1.append(idx01 + 4*f0Idx)
            if idx11 + 4*f1Idx not in l1:
                l1.append(idx11 + 4*f1Idx)

        assert(len(matrixRows) == len(rhRows))
        assert(len(matrixRows) == 2 * len(quadFacePairsBySharedEdge))
        # solve for the depth factors 2, 3...m
        # print("matrixRows")
        # print(matrixRows)

        # print("rhRows")
        # print(rhRows)

        # sanity check: the sets of vertex indices to merge should be disjoint
        for vs in vertsToMergeByOriginalIdx.values():

            for idx in vs:
                assert(0 <= idx < numFaces * 4)

            for vsRef in vertsToMergeByOriginalIdx.values():
                if vs != vsRef:
                    # check that the current sets are disjoint
                    s1 = set(vs)
                    s2 = set(vsRef)
                    assert(len(s1 & s2) == 0)

        if numQuadFaces > 2:
            Q, R = np.linalg.qr(np.array(matrixRows))
            b = Q.T @ np.array(rhRows)
            factors = [1] + list(np.linalg.solve(R, b).T[0])
        elif numQuadFaces == 2:
            # TODO: special case to work around a bug
            # in the least squares solver that causes
            # an infinte recursion. should be fixed in
            # the solver ideally
            f = 0.5 * (rhRows[0][0] / matrixRows[0][0] + rhRows[1][0] / matrixRows[1][0])
            factors = [1, f]
        elif numQuadFaces == 1:
            factors = [1]

        # print("factors", factors)
        # multiply depths by the factors computed per face depth factors
        # quads = []
        for face in computedCoordsByFace.keys():
            quad = computedCoordsByFace[face]
            idx = indexOfFace(faces, face)
            depthScale = factors[idx]
            for i in range(4):
                vert = quad[i].to_3d()
                # print("vert before", vert)
                vert *= depthScale
                # print("vert after", vert)
                quad[i] = vert
            # quads.append(quad)

        # create the actual blender mesh
        bpy.ops.object.mode_set(mode='OBJECT')
        name = inputMesh.name + '_3D'
        # print("createM  esh", points)
        me = bpy.data.meshes.new(name)
        ob = bpy.data.objects.new(name, me)
        ob.show_name = True
        # Link object to scene
        context.scene.objects.link(ob)
        verts = []
        faces = []
        idx = 0
        for quad in quads:
            quadIdxs = []
            for vert in quad:
                verts.append(vert)
                quadIdxs.append(idx)
                idx += 1
            faces.append(quadIdxs)

        #
        # Step 2:
        #     optionally merge vertices
        #

        # print("in faces", [f.vertices[:] for f in self.mesh.data.faces])
        # print("in edges", [e.vertices[:] for e in self.mesh.data.edges])
        # print("out verts", verts)
        # print("out faces", faces)
        # print("out edges", [e.vertices[:] for e in me.edges])
        # print("vertsToMergeByOriginalIdx")
        # print(vertsToMergeByOriginalIdx)
        if mergeVertices:
            for vs in vertsToMergeByOriginalIdx.values():
                print("merging verts", vs)
                # merge at the mean position, which is guaranteed to
                # lie on the line of sight, since all the vertices do

                mean = Vector((0, 0, 0))
                for idx in vs:
                    # print("idx", idx)
                    # print("verts", verts)
                    currVert = verts[idx]
                    mean += currVert
                mean /= len(vs)

                for idx in vs:
                    verts[idx] = mean
        print("2")
        # Update mesh with new data
        me.from_pydata(verts, [], faces)
        me.update(calc_edges=True)
        ob.select = True
        context.scene.objects.active = ob

        # finally remove doubles
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')

        return ob

    def getOutputMeshScale(self, camera, inMesh, outMesh):
        inMeanPos = Vector((0.0, 0.0, 0.0))
        cmi = camera.matrix_world.inverted()
        mm = inMesh.matrix_world
        for v in inMesh.data.vertices:
            vCamSpace = (cmi * mm * v.co.to_4d()).to_3d()
            inMeanPos += vCamSpace
        inMeanPos /= len(inMesh.data.vertices)

        outMeanPos = Vector((0.0, 0.0, 0.0))
        for v in outMesh.data.vertices:
            outMeanPos += v.co
        outMeanPos /= len(outMesh.data.vertices)

        inDistance = inMeanPos.length
        outDistance = outMeanPos.length

        print(inMeanPos, outMeanPos, inDistance, outDistance)

        if outDistance == 0.0:
            # if we need to handle this case, we probably have bigger problems...
            # anyway, return 1.
            return 1

        return inDistance / outDistance

    def areAllMeshFacesConnected(self, mesh):

        def getFaceNeighbors(faces, face):
            neighbors = []
            for v in face.vertices:
                for otherFace in faces:
                    if otherFace == face:
                        continue
                    if len(set(face.vertices) & set(otherFace.vertices)) == 2:
                        # these faces share an edge
                        if otherFace not in neighbors:
                            neighbors.append(otherFace)
            return neighbors

        def visitNeighbors(faces, face, visitedFaces):
            ns = getFaceNeighbors(faces, face)
            for n in ns:
                if n not in visitedFaces:
                    visitedFaces.append(n)
                    visitNeighbors(faces, n, visitedFaces)

        faces = mesh.data.polygons
        if len(faces) == 1:
            return True
        visitedFaces = []
        visitNeighbors(faces, faces[0], visitedFaces)

        return len(visitedFaces) == len(faces)

    def areAllMeshFacesQuads(self, mesh):
        for f in mesh.data.polygons:
            if len(f.vertices) != 4:
                return False
        return True

    def execute(self, context):
        props = bpy.context.scene.blam

        #
        # get the active camera
        #
        self.camera = context.scene.camera
        if self.camera is None:
            self.report({'ERROR'}, "There is no active camera")

        #
        # get the mesh containing the quads, assume it's the active object
        #
        self.mesh = context.active_object

        if self.mesh is None:
            self.report({'ERROR'}, "There is no active object")
            return {'CANCELLED'}
        if 'Mesh' not in str(type(self.mesh.data)):
            self.report({'ERROR'}, "The active object is not a mesh")
            return {'CANCELLED'}
        if len(self.mesh.data.polygons) == 0:
            self.report({'ERROR'}, "The mesh does not have any faces.")
            return {'CANCELLED'}
        if not self.areAllMeshFacesQuads(self.mesh):
            self.report({'ERROR'}, "The mesh must consist of quad faces only.")
            return {'CANCELLED'}
        if not self.areAllMeshFacesConnected(self.mesh):
            self.report({'ERROR'}, "All faces of the input mesh must be connected.")
            return {'CANCELLED'}

        #
        # process all quads from the mesh individually, computing vertex depth
        # values for each of them.
        #
        # a dictionary associating computed quads with faces in the original mesh.
        # used later when building the final output mesh
        computedCoordsByFace = {}
        quads = []
        # loop over all faces
        for f in self.mesh.data.polygons:
            # if this is a quad face, process it
            if len(f.vertices) == 4:
                assert(f not in computedCoordsByFace.keys())

                # gather quad vertices in the local coordinate frame of the
                # mesh
                inputPointsLocalMeshSpace = []
                for idx in f.vertices:
                    inputPointsLocalMeshSpace.append(self.mesh.data.vertices[idx])

                # transform vertices to camera space
                inputPointsCameraSpace = self.worldToCameraSpace(inputPointsLocalMeshSpace)

                # compute normalized input vectors (eq 16)
                qHats = [x.normalized() for x in inputPointsCameraSpace]

                # run the algorithm to create a quad with depth. coords in camera space
                outputPointsCameraSpace = self.computeQuadDepthInformation(*qHats)

                # store the index in the original mesh of the computed quad verts.
                # used later when constructing the output mesh.
                # print("quad")
                for i in range(4):
                    outputPointsCameraSpace[i].resize_4d()
                    outputPointsCameraSpace[i].w = f.vertices[i]  # w is an index, not a coordinate

                # remember which original mesh face corresponds to the computed quad
                computedCoordsByFace[f] = outputPointsCameraSpace
                quads.append(outputPointsCameraSpace)
            else:
                assert(False)  # no non-quads allowed. should have been caught earlier

        m = self.createMesh(context, self.mesh, computedCoordsByFace, quads, not props.separate_faces)

        # up intil now, coords have been in camera space. transform the final mesh so
        # its transform (and thus origin) conicides with the camera.
        m.matrix_world = context.scene.camera.matrix_world

        # finally apply a uniform scale that matches the distance between
        # the camera and mean point of the two meshes
        uniformScale = self.getOutputMeshScale(self.camera, self.mesh, m)
        m.scale = Vector((uniformScale,) * 3)

        return {'FINISHED'}


class BLAM_PT_camera_calibration(bpy.types.Panel):
    '''The GUI for the focal length and camera orientation functionality.'''
    bl_label = "Static Camera Calibration"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"

    def draw(self, context):
        layout = self.layout
        props = context.scene.blam

        layout.prop(props, "calibration_type")

        row = layout.row()
        box = row.box()
        box.label(text="1st Vanishing Point")
        box.prop(props, "vp1_axis", text="Parallel to")

        row = layout.row()
        box = row.box()
        singleVp = props.calibration_type == 'ONE_VP'
        if singleVp:
            box.label(text="Horizon line")
            box.prop(props, "use_horizon_segment")
            # box.label(text="An optional single line segment parallel to the horizon.")
        else:
            box.label(text="2nd Vanishing Point")
            box.prop(props, "vp2_axis", text="Parallel to")

        row = layout.row()
        # row.enabled = singleVp
        if singleVp:
            row.prop(props, "up_axis")
        else:
            row.prop(props, "optical_center_type")
        # TODO layout.prop(props, "vp1_only")

        layout.operator("blam.setup_grease_pencil_layers", icon='GREASEPENCIL')
        layout.prop(props, "set_cambg")
        layout.operator("blam.calibrate_active_camera", icon='CAMERA_DATA')


class BLAM_OT_calibrate_active_camera(bpy.types.Operator):
    '''\brief This operator handles estimation of focal length and camera orientation
    from input line segments. All sections numbers, equations numbers etc
    refer to "Using Vanishing Points for Camera Calibration and Coarse 3D Reconstruction
    from a Single Image" by E. Guillou, D. Meneveaux, E. Maisel, K. Bouatouch.
    (http://www.irisa.fr/prive/kadi/Reconstruction/paper.ps.gz).
    '''
    bl_idname = "blam.calibrate_active_camera"
    bl_label = "Calibrate Active Camera"
    bl_description = "Computes the focal length and orientation of the active camera based on " \
                     "the provided grease pencil strokes"
    bl_options = {'REGISTER', 'UNDO'}

    def computeSecondVanishingPoint(self, Fu, f, P, horizonDir):
        '''Computes the coordinates of the second vanishing point
        based on the first, a focal length, the center of projection and
        the desired horizon tilt angle. The equations here are derived from
        section 3.2 "Determining the focal length from a single image".
        :param Fu: the first vanishing point in normalized image coordinates.
        :param f: the relative focal length.
        :param P: the center of projection in normalized image coordinates.
        :param horizonDir: The desired horizon direction.
        :return: The coordinates of the second vanishing point.
        '''

        # find the second vanishing point
        # TODO 1: take principal point into account here
        # TODO 2: if the first vanishing point coincides with the image center,
        #        these lines won't work, but this case should be handled somehow.
        k = -(Fu.length_squared + f**2) / Fu.dot(horizonDir)
        Fv = Fu + k * horizonDir

        return Fv

    def computeFocalLength(self, Fu, Fv, P):
        '''Computes the focal length based on two vanishing points and a center of projection.
        See 3.2 "Determining the focal length from a single image"
        :param Fu: the first vanishing point in normalized image coordinates.
        :param Fv: the second vanishing point in normalized image coordinates.
        :param P: the center of projection in normalized image coordinates.
        :return: The relative focal length.
        '''

        # compute Puv, the orthogonal projection of P onto FuFv
        dirFuFv = (Fu - Fv).normalized()
        FvP = P - Fv
        proj = dirFuFv.dot(FvP)
        Puv = proj * dirFuFv + Fv

        PPuv = (P - Puv).length

        FvPuv = (Fv - Puv).length
        FuPuv = (Fu - Puv).length
        # FuFv = (Fu - Fv).length
        # print("FuFv", FuFv, "FvPuv + FuPuv", FvPuv + FuPuv)

        fSq = FvPuv * FuPuv - PPuv * PPuv
        # print("FuPuv", FuPuv, "FvPuv", FvPuv, "PPuv", PPuv, "OPuv", FvPuv * FuPuv)
        # print("fSq = ", fSq, " = ", FvPuv * FuPuv, " - ", PPuv * PPuv)
        if fSq < 0:
            return None
        f = sqrt(fSq)
        # print("dot 1:", Vector(list[Fu] + [f]).normalized().dot(Vector(list[Fv] + [f]).normalized())

        return f

    def computeCameraRotationMatrix(self, Fu, Fv, f, P):
        '''Computes the camera rotation matrix based on two vanishing points
        and a focal length as in section 3.3 "Computing the rotation matrix".
        :param Fu: the first vanishing point in normalized image coordinates.
        :param Fv: the second vanishing point in normalized image coordinates.
        :param f: the relative focal length.
        :return: The matrix Moc
        '''
        Fu -= P
        Fv -= P

        OFu = Vector((Fu.x, Fu.y, f))
        OFv = Vector((Fv.x, Fv.y, f))

        # print("matrix dot", OFu.dot(OFv))

        upRc = OFu.normalized()
        vpRc = OFv.normalized()
        wpRc = upRc.cross(vpRc)

        M = Matrix((upRc, vpRc, wpRc)).to_4x4()

        return M

    def alignCoordinateAxes(self, M, ax1, ax2):
        '''Modifies the original camera transform to make the coordinate axes line
        up as specified.
        :param M: the original camera rotation matrix
        :param ax1: The index of the axis to align with the first layer segments.
        :param ax2: The index of the axis to align with the second layer segments.
        :return: The final camera rotation matrix.
        '''
        # line up the axes as specified in the ui
        x180Rot = Euler((radians(180.0), 0, 0), 'XYZ').to_matrix().to_4x4()
        z180Rot = Euler((0, 0, radians(180.0)), 'XYZ').to_matrix().to_4x4()
        zn90Rot = Euler((0, 0, radians(-90.0)), 'XYZ').to_matrix().to_4x4()
        xn90Rot = Euler((radians(-90.0), 0, 0), 'XYZ').to_matrix().to_4x4()
        y90Rot = Euler((0, radians(90.0), 0), 'XYZ').to_matrix().to_4x4()

        M = x180Rot * M * z180Rot

        # vp1 vp2 up
        # ----------
        # -x, +y, +z  rules to determine the directions of the axes:
        # +y, +x, +z   * up-axis always should be positive
        # -x, -z, +y   * if y axis is not up-axis, it should be positive
        # -z, +x, +y   * if y axis is up-axis, z axis shold be negative
        # +y, -z, +x
        # +z, +y, +x

        if ax1 == 0 and ax2 == 1:
            # print("x, y")
            pass
        elif ax1 == 1 and ax2 == 0:
            # print("y, x")
            M = zn90Rot * M
        elif ax1 == 0 and ax2 == 2:
            # print("x, z")
            M = xn90Rot * M
        elif ax1 == 2 and ax2 == 0:
            # print("z, x")
            M = xn90Rot * zn90Rot * M
        elif ax1 == 1 and ax2 == 2:
            # print("y, z")
            M = y90Rot * zn90Rot * M
        elif ax1 == 2 and ax2 == 1:
            # print("z, y")
            M = y90Rot * M

        return M

    def gatherGreasePencilSegments(self, gpl):
        '''Collects and returns line segments in normalized image coordinates
        from the first two grease pencil layers.
        :param gpl: A collection of grease pencil layers
        :return: A list of line segment sets. [i][j][k][l] is coordinate l of point k
        in segment j from layer i.
        '''

        # loop over grease pencil layers and gather line segments
        vpLineSets = []
        for layer in gpl:
            if not layer.active_frame:
                # ignore empty layers
                continue
            strokes = layer.active_frame.strokes
            lines = []
            for s in strokes:
                if len(s.points) == 2:
                    # this is a line segment. add it.
                    line = [p.co.to_2d() for p in s.points]
                    lines.append(line)

            vpLineSets.append(lines)
        return vpLineSets

    def computeIntersectionPointForLineSegments(self, lineSet):
        '''Computes the intersection point in a least squares sense of
        a collection of line segments.
        '''
        matrixRows = []
        rhsRows = []

        for line in lineSet:
            # a point on the line
            p = line[0]
            # a unit vector parallel to the line
            dir = (line[1] - line[0]).normalized()
            # a unit vector perpendicular to the line
            n = dir.orthogonal()
            matrixRows.append(n)
            rhsRows.append([p.dot(n)])

        Q, R = np.linalg.qr(np.array(matrixRows))
        b = Q.T @ np.array(rhsRows)
        vp = Vector(np.linalg.solve(R, b))
        return vp

    def computeTriangleOrthocenter(self, verts):
        # print("verts", verts)
        assert(len(verts) == 3)

        A = verts[0]
        B = verts[1]
        C = verts[2]

        # print("A, B, C", A, B, C)

        a, b = A
        c, d = B
        e, f = C

        N = b*c + d*e + f*a - c*f - b*e - a*d
        x = ((d-f)*b*b + (f-b)*d*d + (b-d)*f*f + a*b*(c-e) + c*d*(e-a) + e*f*(a-c)) / N
        y = ((e-c)*a*a + (a-e)*c*c + (c-a)*e*e + a*b*(f-d) + c*d*(b-f) + e*f*(d-b)) / N

        return Vector((x, y))

    def imgAspect(self, imageWidth, imageHeight, sensor_fit):
        if sensor_fit == 'AUTO' and imageWidth >= imageHeight or sensor_fit == 'HORIZONTAL':
            return (1, imageHeight / imageWidth)
        else:
            return (imageWidth / imageHeight, 1)

    def relImgCoords2ImgPlaneCoords(self, pt, imageWidth, imageHeight, sensor_fit):
        sw, sh = self.imgAspect(imageWidth, imageHeight, sensor_fit)
        return Vector((sw * (pt[0] - 0.5), sh * (pt[1] - 0.5)))

    def execute(self, context):
        '''Executes the operator.
        :param context: The context in which the operator was executed.
        '''
        props = context.scene.blam

        singleVp = props.calibration_type == 'ONE_VP'
        useHorizonSegment = props.use_horizon_segment
        setBgImg = props.set_cambg

        #
        # get the active camera
        #
        cam = context.scene.camera
        if not cam:
            self.report({'ERROR'}, "No active camera.")
            return {'CANCELLED'}

        #
        # check settings
        #
        if singleVp:
            upAxisIndex = ['X', 'Y', 'Z'].index(props.up_axis)
            vp1AxisIndex = ['X', 'Y', 'Z'].index(props.vp1_axis)

            if upAxisIndex == vp1AxisIndex:
                self.report({'ERROR'}, "The up axis cannot be parallel to the axis pointing to the vanishing point.")
                return {'CANCELLED'}
            vp2AxisIndex = (set([0, 1, 2]) - set([upAxisIndex, vp1AxisIndex])).pop()
            vpAxisIndices = [vp1AxisIndex, vp2AxisIndex]
        else:
            vp1AxisIndex = ['X', 'Y', 'Z'].index(props.vp1_axis)
            vp2AxisIndex = ['X', 'Y', 'Z'].index(props.vp2_axis)
            vpAxisIndices = [vp1AxisIndex, vp2AxisIndex]
            setBgImg = props.set_cambg

            if vpAxisIndices[0] == vpAxisIndices[1]:
                self.report({'ERROR'}, "The two different vanishing points cannot be computed from the same axis.")
                return {'CANCELLED'}

        #
        # gather lines for each vanishing point
        #
        activeSpace = context.space_data

        if not activeSpace.clip:
            self.report({'ERROR'}, "There is no active movie clip.")
            return {'CANCELLED'}

        # check that we have the number of layers we need
        if not activeSpace.clip.grease_pencil:
            self.report({'ERROR'}, "There is no grease pencil datablock.")
            return {'CANCELLED'}
        gpl = activeSpace.clip.grease_pencil.layers
        if len(gpl) == 0:
            self.report({'ERROR'}, "There are no grease pencil layers.")
            return {'CANCELLED'}
        if len(gpl) < 2 and not singleVp:
            self.report({'ERROR'}, "Calibration using two vanishing points requires two grease pencil layers.")
            return {'CANCELLED'}
        if len(gpl) < 2 and singleVp and useHorizonSegment:
            self.report(
                {'ERROR'},
                "Single vanishing point calibration with a custom horizon line requires two grease pencil layers")
            return {'CANCELLED'}

        vpLineSets = self.gatherGreasePencilSegments(gpl)

        # check that we have the expected number of line segment strokes
        if len(vpLineSets[0]) < 2:
            self.report({'ERROR'}, "The first grease pencil layer must contain at least two line segment strokes.")
            return {'CANCELLED'}
        if not singleVp and len(vpLineSets[1]) < 2:
            self.report({'ERROR'}, "The second grease pencil layer must contain at least two line segment strokes.")
            return {'CANCELLED'}
        if singleVp and useHorizonSegment and len(vpLineSets[1]) != 1:
            self.report(
                {'ERROR'},
                "The second grease pencil layer must contain exactly one line segment stroke (the horizon line).")
            return {'CANCELLED'}

        #
        # get the principal point P in image plane coordinates
        # TODO: get the value from the camera data panel,
        # currently always using the image center
        #
        imageWidth = activeSpace.clip.size[0]
        imageHeight = activeSpace.clip.size[1]
        sf = cam.data.sensor_fit

        # principal point in image plane coordinates.
        # in the middle of the image by default
        P = Vector((0, 0))

        if singleVp:
            #
            # calibration using a single vanishing point
            #
            # compute the horizon direction
            horizDir = Vector((1.0, 0.0)).normalized()  # flat horizon by default
            if useHorizonSegment:
                ax, ay = self.imgAspect(imageWidth, imageHeight, sf)
                xHorizDir, yHorizDir = vpLineSets[1][0][1] - vpLineSets[1][0][0]
                horizDir = Vector((-ax * xHorizDir, -ay * yHorizDir)).normalized()
            # print("horizDir", horizDir)

            # compute the vanishing point location
            vp1 = self.computeIntersectionPointForLineSegments(vpLineSets[0])

            # get the current relative focal length
            fAbs = activeSpace.clip.tracking.camera.focal_length
            sensorWidth = activeSpace.clip.tracking.camera.sensor_width

            f = fAbs / sensorWidth
            # print("fAbs", fAbs, "f rel", f)
            Fu = self.relImgCoords2ImgPlaneCoords(vp1, imageWidth, imageHeight, sf)
            Fv = self.computeSecondVanishingPoint(Fu, f, P, horizDir)

            # order vanishing points along the image x axis
            if Fv.x < Fu.x:
                Fu, Fv = Fv, Fu
                vpAxisIndices.reverse()
        else:
            #
            # calibration using two vanishing points
            #
            if props.optical_center_type == 'CAMDATA':
                # get the principal point location from camera data
                P = Vector(activeSpace.clip.tracking.camera.principal)
                # print("camera data optical center", P[:])
                P.x /= imageWidth
                P.y /= imageHeight
                # print("normlz. optical center", P[:])
                P = self.relImgCoords2ImgPlaneCoords(P, imageWidth, imageHeight, sf)
            elif props.optical_center_type == 'COMPUTE':
                if len(vpLineSets) < 3:
                    self.report({'ERROR'}, "A third grease pencil layer is needed to compute the optical center.")
                    return {'CANCELLED'}
                # compute the principal point using a vanishing point from a third gp layer.
                # this computation does not rely on the order of the line sets
                vps = [self.computeIntersectionPointForLineSegments(ls) for ls in vpLineSets]
                vps = [self.relImgCoords2ImgPlaneCoords(vp, imageWidth, imageHeight, sf) for vp in vps]
                P = self.computeTriangleOrthocenter(vps)
            else:
                # assume optical center in image midpoint
                pass

            # compute the two vanishing points
            vps = [self.computeIntersectionPointForLineSegments(vpLineSets[i]) for i in range(2)]
            Fu = self.relImgCoords2ImgPlaneCoords(vps[0], imageWidth, imageHeight, sf)
            Fv = self.relImgCoords2ImgPlaneCoords(vps[1], imageWidth, imageHeight, sf)

            # order vanishing points along the image x axis
            if Fv.x < Fu.x:
                Fu, Fv = Fv, Fu
                vpAxisIndices.reverse()

            #
            # compute focal length
            #
            f = self.computeFocalLength(Fu, Fv, P)

            if f is None:
                self.report({'ERROR'}, "Failed to compute focal length. Invalid vanishing point constellation.")
                return {'CANCELLED'}

        #
        # compute camera orientation
        #
        print(Fu, Fv, f)
        # initial orientation based on the vanishing points and focal length
        M = self.computeCameraRotationMatrix(Fu, Fv, f, P)

        # sanity check: M should be a pure rotation matrix,
        # so its determinant should be 1
        eps = 0.00001
        if abs(1.0 - M.determinant()) > eps:
            self.report({'ERROR'}, "Non unit rotation matrix determinant: " + str(M.determinant()))
            # return {'CANCELLED'}

        # align the camera to the coordinate axes as specified
        M = self.alignCoordinateAxes(M, vpAxisIndices[0], vpAxisIndices[1])
        # apply the transform to the camera
        cam.matrix_world = M

        #
        # move the camera an arbitrary distance away from the ground plane
        # TODO: focus on the origin or something
        #
        cam.location = (0, 0, 2)

        # compute an absolute focal length in mm based
        # on the current camera settings
        if cam.data.sensor_fit == 'VERTICAL':
            fMm = cam.data.sensor_height * f
        else:
            fMm = cam.data.sensor_width * f
        cam.data.lens = fMm
        self.report({'INFO'}, "Camera focal length set to " + str(fMm))

        # move principal point of the blender camera
        cam.data.shift_x = -P.x
        cam.data.shift_y = -P.y

        #
        # set the camera background image
        #
        context.scene.render.resolution_x = imageWidth
        context.scene.render.resolution_y = imageHeight

        if setBgImg:
            bpy.ops.clip.set_viewport_background()

        return {'FINISHED'}


class BLAM_OT_setup_grease_pencil_layers(bpy.types.Operator):
    bl_idname = "blam.setup_grease_pencil_layers"
    bl_label = "Setup Grease Pencil Layers"
    bl_description = "Setup Grease Pencil layers according to parameters of the camera calibration tool"
    bl_options = {'REGISTER', 'UNDO'}

    axisColors = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}

    def axisName(self, axis):
        return (axis, "{} Axis".format(axis))

    def execute(self, context):
        props = context.scene.blam
        activeSpace = context.space_data

        if not activeSpace.clip:
            self.report({'ERROR'}, "There is no active movie clip.")
            return {'CANCELLED'}

        axisNames = [self.axisName(props.vp1_axis)]

        if props.calibration_type == 'ONE_VP' and props.use_horizon_segment:
            if props.vp1_axis == props.up_axis:
                self.report({'ERROR'}, "The up axis cannot be parallel to the axis pointing to the vanishing point.")
                return {'CANCELLED'}
            axis = (set(['X', 'Y', 'Z']) - set([props.vp1_axis, props.up_axis])).pop()
            axisNames.append((axis, "Horizon"))
        elif props.calibration_type == 'TWO_VP':
            if props.vp1_axis == props.vp2_axis:
                self.report({'ERROR'}, "The two different vanishing points cannot be computed from the same axis.")
                return {'CANCELLED'}
            axisNames.append(self.axisName(props.vp2_axis))
            if props.optical_center_type == 'COMPUTE':
                axis = (set(['X', 'Y', 'Z']) - set([props.vp1_axis, props.vp2_axis])).pop()
                axisNames.append(self.axisName(axis))

        activeSpace.grease_pencil_source = 'CLIP'
        context.scene.tool_settings.gpencil_stroke_placement_view2d = 'CURSOR'
        activeSpace.show_grease_pencil = True

        if not activeSpace.clip.grease_pencil:
            bpy.ops.gpencil.data_add()

        gpl = activeSpace.clip.grease_pencil.layers

        for layer in gpl:
            gpl.remove(layer)

        for axis, name in axisNames:
            layer = gpl.new(name, set_active=True)
            layer.tint_color = self.axisColors[axis]
            layer.tint_factor = 1.0

        return {'FINISHED'}


class BLAMProps(bpy.types.PropertyGroup):

    # Focal length and orientation estimation stuff

    calibration_type = bpy.props.EnumProperty(
        name="Method",
        description="The type of calibration method to use",
        items=[('ONE_VP', "One Vanishing Point",
                "Estimates the camera orientation using a known focal length, "
                "a single vanishing point and an optional horizon tilt angle"),
               ('TWO_VP', "Two Vanishing Points",
                "Estimates the camera focal length and orientation from two vanishing points")],
        default=('TWO_VP'))

    vp1_axis = bpy.props.EnumProperty(
        name="Axis 1",
        description="The axis to which the line segments from the first grease pencil layer are parallel in 3D space",
        items=[('X', "X Axis", "Consider line segments in 1st grease pencil layer are parallel to X axis"),
               ('Y', "Y Axis", "Consider line segments in 1st grease pencil layer are parallel to Y axis"),
               ('Z', "Z Axis", "Consider line segments in 1st grease pencil layer are parallel to Z axis")],
        default=('X'))

    vp2_axis = bpy.props.EnumProperty(
        name="Axis 2",
        description="The axis to which the line segments from the second grease pencil layer are parallel in 3D space",
        items=[('X', "X Axis", "Consider line segments in 2nd grease pencil layer are parallel to X axis"),
               ('Y', "Y Axis", "Consider line segments in 2nd grease pencil layer are parallel to Y axis"),
               ('Z', "Z Axis", "Consider line segments in 2nd grease pencil layer are parallel to Z axis")],
        default=('Y'))

    up_axis = bpy.props.EnumProperty(
        name="Up Axis",
        description="The up axis for single vanishing point calibration",
        items=[('X', "X Axis", "Use X axis as up axis"),
               ('Y', "Y Axis", "Use Y axis as up axis"),
               ('Z', "Z Axis", "Use Z axis as up axis")],
        default=('Z'))

    optical_center_type = bpy.props.EnumProperty(
        name="Optical Center",
        description="How the optical center is computed for calibration using two vanishing points",
        items=[('MID', "Image Midpoint",
                "Assume the optical center coincides with the image midpoint (reasonable in most cases)"),
               ('CAMDATA', "From Camera Data",
                "Get a known optical center from the current camera data"),
               ('COMPUTE', "From 3rd Vanishing Point",
                "Computes the optical center using a third vanishing point from grease pencil layer 3")],
        default=('MID'))

    # vp1_only = bpy.props.BoolProperty(
    #     name="Only use first line set",
    #     description="",
    #     default=False)

    set_cambg = bpy.props.BoolProperty(
        name="Set Background Image",
        description="Automatically set the current movie clip as the camera background image "
                    "when performing camera calibration (works only when a 3D view-port is visible)",
        default=True)

    use_horizon_segment = bpy.props.BoolProperty(
        name="Compute from grease pencil stroke",
        description="Extract the horizon angle from a single line segment in the second grease pencil layer. "
                    "If unchecked, the horizon angle is set to 0",
        default=True)

    # 3D reconstruction stuff

    separate_faces = bpy.props.BoolProperty(
        name="Separate Faces",
        description="Do not join the faces in the reconstructed mesh. Useful for finding problematic faces",
        default=False)

    projection_method = bpy.props.EnumProperty(
        name="Method",
        description="The method to use to project the image onto the mesh",
        items=[('SIMPLE', "Simple",
                "Uses UV coordinates projected from the camera view. May give warping on large faces"),
               ('HQ', "High Quality",
                "Uses a UV Project modifier combined with a simple subdivision surface modifier")],
        default=('HQ'))


classes = (
    BLAMProps,
    BLAM_PT_photo_modeling_tools,
    BLAM_OT_set_pivot_to_camera,
    BLAM_OT_project_bg_onto_mesh,
    BLAM_OT_reconstruct_mesh_with_rects,
    BLAM_PT_camera_calibration,
    BLAM_OT_calibrate_active_camera,
    BLAM_OT_setup_grease_pencil_layers,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.blam = bpy.props.PointerProperty(type=BLAMProps)


def unregister():
    del bpy.types.Scene.blam

    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
