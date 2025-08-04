from  abaqus import*
from  caeModules import *
from  abaqusConstants import*
from  visualization import *
from  odbAccess import *
from  abaqusConstants import *
from  textRepr import *
from  odbSection import *
import sys
sys.path.append(r'E:\program_file\anaconda\envs\abaqus_py\Lib\site-packages')
import numpy as np


# 读取裂缝坐标CSV 文件（假设数据是数值型的）
data = np.loadtxt(r'F:\python_project\Simulation\abaqus_models\rectangles_crack.csv', delimiter=',', skiprows=1)  # 如果有表头，可以跳过
num_rows=data.shape[0]

#批量化建模
for i in range(1,240):
    x1,y1,x2,y2,x3,y3,x4,y4=data[i-200][0],data[i-200][1],data[i-200][2],data[i-200][3],data[i-200][4],data[i-200][5],data[i-200][6],data[i-200][7]#读取裂缝坐标
    #初始化模型
    model_name = f'Model-{i + 1}'
    Job_name = f'Job-{i + 1}'
    #: 模型 "Model-i" 已创建.
    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)

    #: 新的模型数据库已创建.
    #: 模型 "Model-i" 已创建.
    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    s = mdb.models[model_name].ConstrainedSketch(name='__profile__',
                                                sheetSize=1000.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.rectangle(point1=(0.0, 0.0), point2=(400.0, 400.0))
    p = mdb.models[model_name].Part(name='slab', dimensionality=THREE_D,
                                   type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['slab']
    p.BaseSolidExtrude(sketch=s, depth=50.0)
    s.unsetPrimaryObject()
    p = mdb.models[model_name].parts['slab']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']
    s1 = mdb.models[model_name].ConstrainedSketch(name='__profile__',
                                                 sheetSize=50.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(0.0, 10.0))
    p = mdb.models[model_name].Part(name='PZT', dimensionality=THREE_D,
                                   type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['PZT']
    p.BaseSolidExtrude(sketch=s1, depth=2.0)
    s1.unsetPrimaryObject()
    p = mdb.models[model_name].parts['PZT']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
                                                           engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    mdb.models[model_name].Material(name='concrete')
    mdb.models[model_name].materials['concrete'].Density(table=((2.5e-09,),))
    mdb.models[model_name].materials['concrete'].Elastic(table=((30000.0, 0.2),))
    mdb.models[model_name].Material(name='PZT')
    mdb.models[model_name].materials['PZT'].Density(table=((7.5e-09,),))
    mdb.models[model_name].materials['PZT'].Elastic(type=ENGINEERING_CONSTANTS,
                                                   table=(
                                                   (60610.0, 60610.0, 48310.0, 0.289, 0.512, 0.512, 23500.0, 23000.0,
                                                    23000.0),))
    mdb.models[model_name].materials['PZT'].Dielectric(type=ORTHOTROPIC, table=((
                                                                                   1.505e-14, 1.505e-14, 1.301e-14),))
    mdb.models[model_name].materials['PZT'].Piezoelectric(type=STRAIN, table=((0.0,
                                                                              0.0, 0.0, 7.41e-10, 0.0, 0.0, 0.0, 0.0,
                                                                              0.0, 0.0, 0.0, 7.41e-10, -2.74e-10,
                                                                              -2.74e-10, 5.93e-10, 0.0, 0.0, 0.0),))
    mdb.models[model_name].HomogeneousSolidSection(name='PZT', material='PZT',
                                                  thickness=None)
    mdb.models[model_name].HomogeneousSolidSection(name='concrete',
                                                  material='concrete', thickness=None)
    mdb.models[model_name].materials.changeKey(fromName='PZT', toName='PZT-5H')
    p = mdb.models[model_name].parts['PZT']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    region = p.Set(cells=cells, name='pzt')
    p = mdb.models[model_name].parts['PZT']
    p.SectionAssignment(region=region, sectionName='PZT', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    p = mdb.models[model_name].parts['slab']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['slab']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    region = p.Set(cells=cells, name='slab')
    p = mdb.models[model_name].parts['slab']
    p.SectionAssignment(region=region, sectionName='concrete', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    p = mdb.models[model_name].parts['PZT']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    #: 警告: 无法继续--请完成该步骤或取消这一操作.
    session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=49.0201,
                                                    farPlane=67.6744, width=50.6881, height=22.2224,
                                                    cameraPosition=(14.5883,
                                                                    -14.0157, 55.7318),
                                                    cameraUpVector=(-0.512075, 0.858913, -0.00688767))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=43.693,
                                                    farPlane=74.0864, width=45.1798, height=19.8075,
                                                    cameraPosition=(55.8175,
                                                                    -9.52913, 17.1782),
                                                    cameraUpVector=(-0.534811, 0.302405, 0.789005),
                                                    cameraTarget=(0.339997, -0.104225, 0.729901))
    p = mdb.models[model_name].parts['PZT']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    region = regionToolset.Region(cells=cells)
    orientation = None
    mdb.models[model_name].parts['PZT'].MaterialOrientation(region=region,
                                                           orientationType=SYSTEM, axis=AXIS_1, localCsys=orientation,
                                                           fieldName='',
                                                           additionalRotationType=ROTATION_NONE, angle=0.0,
                                                           additionalRotationField='', stackDirection=STACK_3)
    #: 指定的材料方向已经被指派到所选区域.
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models[model_name].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models[model_name].parts['PZT']
    a.Instance(name='PZT-1', part=p, dependent=ON)
    p = mdb.models[model_name].parts['slab']
    a.Instance(name='slab-1', part=p, dependent=ON)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=876.602,
                                                    farPlane=1504.95, width=910.507, height=399.179,
                                                    cameraPosition=(-731.41,
                                                                    585.781, 662.971),
                                                    cameraUpVector=(0.150497, 0.613504, -0.775218),
                                                    cameraTarget=(206.116, 192.925, 15.9595))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=949.759,
                                                    farPlane=1431.12, width=986.495, height=432.493,
                                                    cameraPosition=(-243.47,
                                                                    626.952, 1043.98),
                                                    cameraUpVector=(-0.219961, 0.626957, -0.747357),
                                                    cameraTarget=(200.306, 192.435, 11.4224))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=1002.19,
                                                    farPlane=1377.91, width=1040.96, height=456.369,
                                                    cameraPosition=(-17.7855,
                                                                    503.544, 1154.49),
                                                    cameraUpVector=(-0.232978, 0.764125, -0.601527),
                                                    cameraTarget=(197.554, 193.94, 10.075))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=910.215,
                                                    farPlane=1469.42, width=945.427, height=414.488,
                                                    cameraPosition=(-899.367,
                                                                    184.683, 491.825),
                                                    cameraUpVector=(0.176193, 0.894452, -0.410989),
                                                    cameraTarget=(208.595, 197.933, 18.3741))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=938.141,
                                                    farPlane=1441.49, width=631.898, height=277.033,
                                                    viewOffsetX=-52.1135,
                                                    viewOffsetY=-48.5078)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=949.139,
                                                    farPlane=1561.34, width=639.306, height=280.28,
                                                    cameraPosition=(-765.109,
                                                                    -419.207, -509.04),
                                                    cameraUpVector=(-0.361939, 0.672968, 0.645069),
                                                    cameraTarget=(147.481, 112.534, 70.909), viewOffsetX=-52.7244,
                                                    viewOffsetY=-49.0765)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=981.927,
                                                    farPlane=1528.55, width=191.874, height=84.12, viewOffsetX=6.7077,
                                                    viewOffsetY=-14.2683)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=1127.64,
                                                    farPlane=1586.49, width=220.348, height=96.6034,
                                                    cameraPosition=(-287.947,
                                                                    -637.62, 975.984),
                                                    cameraUpVector=(-0.56113, 0.82565, -0.0586127),
                                                    cameraTarget=(162.542, 95.103, 132.132), viewOffsetX=7.70312,
                                                    viewOffsetY=-16.3857)
    a = mdb.models[model_name].rootAssembly
    a.translate(instanceList=('PZT-1',), vector=(200.0, 200.0, 50.0))
    #: The instance PZT-1 was translated by 200., 200., 50. (相对于装配坐标系)
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=908.601,
                                                    farPlane=1363.68, width=784.081, height=343.751,
                                                    cameraPosition=(-474.207,
                                                                    108.822, 935.916),
                                                    cameraUpVector=(-0.173491, 0.984381, -0.0299093))
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF,
                                                           engineeringFeatures=OFF)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)
    p = mdb.models[model_name].parts['PZT']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['slab']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    p = mdb.models[model_name].parts['slab']
    f, e = p.faces, p.edges
    t = p.MakeSketchTransform(sketchPlane=f[4], sketchUpEdge=e[7],
                              sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(200.0, 200.0,
                                                                                      50.0))
    s = mdb.models[model_name].ConstrainedSketch(name='__profile__',
                                                sheetSize=1135.78, gridSpacing=28.39, transform=t)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models[model_name].parts['slab']
    p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
    s.Line(point1=(x1, y1), point2=(x2,y2))
    s.Line(point1=(x2,y2), point2=(x3,y3))
    s.PerpendicularConstraint(entity1=g[6], entity2=g[7], addUndoState=False)
    s.Line(point1=(x3,y3), point2=(x4,y4))
    s.PerpendicularConstraint(entity1=g[7], entity2=g[8], addUndoState=False)
    s.Line(point1=(x4,y4), point2=(x1, y1))
    s.PerpendicularConstraint(entity1=g[8], entity2=g[9], addUndoState=False)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=1119.23,
                                                    farPlane=1193.52, width=149.693, height=65.6274,
                                                    cameraPosition=(221.888,
                                                                    94.8443, 1181.37),
                                                    cameraTarget=(221.888, 94.8443, 50))
    p = mdb.models[model_name].parts['slab']
    f1, e1 = p.faces, p.edges
    p.CutExtrude(sketchPlane=f1[4], sketchUpEdge=e1[7], sketchPlaneSide=SIDE1,
                 sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
    s.unsetPrimaryObject()
    del mdb.models[model_name].sketches['__profile__']
    session.viewports['Viewport: 1'].view.setValues(nearPlane=841.627,
                                                    farPlane=1429.94, width=725.866, height=318.229,
                                                    cameraPosition=(833.644,
                                                                    -330.661, -754.034),
                                                    cameraUpVector=(0.723128, 0.679216, 0.125505))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=948.296,
                                                    farPlane=1323.27, width=817.863, height=358.562,
                                                    cameraPosition=(749.279,
                                                                    270.903, 1016.6),
                                                    cameraUpVector=(-0.0154455, 0.997907, -0.0627991),
                                                    cameraTarget=(200, 200, 24.9998))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=955.218,
                                                    farPlane=1316.35, width=684.617, height=300.145,
                                                    viewOffsetX=-10.056,
                                                    viewOffsetY=-35.6652)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=889.2,
                                                    farPlane=1381.47, width=637.301, height=279.401,
                                                    cameraPosition=(896.061,
                                                                    -126.227, -810.934),
                                                    cameraUpVector=(0.738796, 0.556482, 0.380144),
                                                    cameraTarget=(217.787, 190.839, 43.126), viewOffsetX=-9.36101,
                                                    viewOffsetY=-33.2003)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=929.039,
                                                    farPlane=1341.63, width=150.816, height=66.1199,
                                                    viewOffsetX=-45.9727,
                                                    viewOffsetY=-62.9932)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=968.294,
                                                    farPlane=1237.36, width=157.189, height=68.9136,
                                                    cameraPosition=(711.068,
                                                                    263.822, -950.759),
                                                    cameraUpVector=(0.78966, 0.44954, 0.417553),
                                                    cameraTarget=(215.108, 188.532, 68.2408), viewOffsetX=-47.9152,
                                                    viewOffsetY=-65.6549)
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=1034.2,
                                                    farPlane=1237.36, width=1009.45, height=442.558,
                                                    viewOffsetX=0.242416,
                                                    viewOffsetY=-11.6964)
    a = mdb.models[model_name].rootAssembly
    a.regenerate()
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=ON)
    mdb.models[model_name].ImplicitDynamicsStep(name='Step-1', previous='Initial',
                                               timePeriod=0.00015, maxNumInc=300, timeIncrementationMethod=FIXED,
                                               initialInc=5e-07, nohaf=OFF, noStop=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    a = mdb.models[model_name].rootAssembly
    f1 = a.instances['slab-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#100 ]',), )
    a.Set(faces=faces1, name='sur_wave')
    #: 集 'sur_wave' 已创建 (1 面).
    mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(variables=(
        'U', 'V', 'A'), frequency=1, region=MODEL, exteriorOnly=OFF,
        sectionPoints=DEFAULT, position=NODES, rebar=EXCLUDE)
    regionDef = mdb.models[model_name].rootAssembly.sets['sur_wave']
    mdb.models[model_name].historyOutputRequests['H-Output-1'].setValues(variables=(
        'U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3', 'V1', 'V2', 'V3', 'VR1', 'VR2',
        'VR3', 'A1', 'A2', 'A3', 'AR1', 'AR2', 'AR3'), frequency=1,
        region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON,
                                                               constraints=ON, connectors=ON, engineeringFeatures=ON,
                                                               adaptiveMeshConstraints=OFF)
    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['slab-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#100 ]',), )
    a.Surface(side1Faces=side1Faces1, name='CP-1-slab-1')
    region1 = a.surfaces['CP-1-slab-1']
    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['PZT-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#4 ]',), )
    a.Surface(side1Faces=side1Faces1, name='CP-1-PZT-1')
    region2 = a.surfaces['CP-1-PZT-1']
    mdb.models[model_name].Tie(name='CP-1-slab-1-PZT-1', main=region1,
                              secondary=region2, positionToleranceMethod=COMPUTED, adjust=ON,
                              constraintEnforcement=SURFACE_TO_SURFACE)
    mdb.models[model_name].constraints['CP-1-slab-1-PZT-1'].swapSurfaces()
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON,
                                                               predefinedFields=ON, interactions=OFF, constraints=OFF,
                                                               engineeringFeatures=OFF)
    mdb.models[model_name].TabularAmplitude(name='five_peaks_120Khz', timeSpan=STEP,
                                           smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (5e-07, 0.000522939), (1e-06,
                                                                                                           0.003884208),
                                                                        (1.5e-06, 0.011524387), (2e-06, 0.022523195),
                                                                        (2.5e-06,
                                                                         0.033393265), (3e-06, 0.038754793),
                                                                        (3.5e-06, 0.032777697), (4e-06,
                                                                                                 0.011058615),
                                                                        (4.5e-06, -0.027547093), (5e-06, -0.079654122),
                                                                        (5.5e-06,
                                                                         -0.137062788), (6e-06, -0.187685815),
                                                                        (6.5e-06, -0.217637716), (7e-06,
                                                                                                  -0.214152236),
                                                                        (7.5e-06, -0.168759232), (8e-06, -0.080026929),
                                                                        (8.5e-06,
                                                                         0.044805424), (9e-06, 0.189809235),
                                                                        (9.5e-06, 0.332172), (1e-05,
                                                                                              0.445669592),
                                                                        (1.05e-05, 0.505283986), (1.1e-05, 0.492158596),
                                                                        (1.15e-05,
                                                                         0.397939953), (1.2e-05, 0.22759282),
                                                                        (1.25e-05, 2.4e-16), (1.3e-05,
                                                                                              -0.253965281),
                                                                        (1.35e-05, -0.495743269),
                                                                        (1.4e-05, -0.68515293), (1.45e-05,
                                                                                                 -0.787229862),
                                                                        (1.5e-05, -0.778641378),
                                                                        (1.55e-05, -0.65249115), (1.6e-05,
                                                                                                  -0.420556781),
                                                                        (1.65e-05, -0.112423309), (1.7e-05, 0.22848732),
                                                                        (1.75e-05,
                                                                         0.551432698), (1.8e-05, 0.806377906),
                                                                        (1.85e-05, 0.952196775), (1.9e-05,
                                                                                                  0.963637384),
                                                                        (1.95e-05, 0.835823455), (2e-05, 0.585467821),
                                                                        (2.05e-05,
                                                                         0.248532834), (2.1e-05, -0.125313443),
                                                                        (2.15e-05, -0.480537491), (2.2e-05,
                                                                                                   -0.76456655),
                                                                        (2.25e-05, -0.936116922),
                                                                        (2.3e-05, -0.971628076), (2.35e-05,
                                                                                                  -0.868738861),
                                                                        (2.4e-05, -0.646259171),
                                                                        (2.45e-05, -0.340698258), (2.5e-05,
                                                                                                   -6.65e-16),
                                                                        (2.55e-05, 0.32439918), (2.6e-05, 0.585813397),
                                                                        (2.65e-05,
                                                                         0.749461803), (2.7e-05, 0.797375941),
                                                                        (2.75e-05, 0.730329041), (2.8e-05,
                                                                                                  0.566598428),
                                                                        (2.85e-05, 0.337927665), (2.9e-05, 0.083523292),
                                                                        (2.95e-05,
                                                                         -0.156779266), (3e-05, -0.348962613),
                                                                        (3.05e-05, -0.469807861), (3.1e-05,
                                                                                                   -0.509654918),
                                                                        (3.15e-05, -0.472632332),
                                                                        (3.2e-05, -0.374520064), (
                                                                            3.25e-05, -0.23882264),
                                                                        (3.3e-05, -0.091910621),
                                                                        (3.35e-05, 0.041809942), (
                                                                            3.4e-05, 0.143826009),
                                                                        (3.45e-05, 0.203914815), (3.5e-05, 0.220727476),
                                                                        (
                                                                            3.55e-05, 0.200650787),
                                                                        (3.6e-05, 0.155365249), (3.65e-05, 0.098733709),
                                                                        (
                                                                            3.7e-05, 0.043725372), (3.75e-05, 4.45e-16),
                                                                        (3.8e-05, -0.027426295), (
                                                                            3.85e-05, -0.038287935),
                                                                        (3.9e-05, -0.036088191),
                                                                        (3.95e-05, -0.026398653),
                                                                        (4e-05, -0.014939595), (4.05e-05, -0.005946692),
                                                                        (4.1e-05, -0.001216183), (
                                                                            4.15e-05, -1.98e-05)))
    session.viewports['Viewport: 1'].assemblyDisplay.hideInstances(instances=(
        'slab-1',))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=812.155,
                                                    farPlane=1165.01, width=700.852, height=307.263,
                                                    cameraPosition=(-782.549,
                                                                    78.981, -416.55),
                                                    cameraUpVector=(-0.0779427, 0.99341, -0.0840346),
                                                    cameraTarget=(218.406, 201.781, 106.732))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=858.289,
                                                    farPlane=1118.88, width=178.469, height=78.2431,
                                                    viewOffsetX=-26.5425,
                                                    viewOffsetY=4.33118)
    a = mdb.models[model_name].rootAssembly
    f1 = a.instances['PZT-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#4 ]',), )
    region = a.Set(faces=faces1, name='pzt_0V')
    mdb.models[model_name].ElectricPotentialBC(name='BC-1', createStepName='Step-1',
                                              region=region, fixed=OFF, distributionType=UNIFORM, fieldName='',
                                              magnitude=0.0, amplitude=UNSET)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=850.875,
                                                    farPlane=1113.79, width=176.927, height=77.5671,
                                                    cameraPosition=(-879.621,
                                                                    74.123, -16.6233),
                                                    cameraUpVector=(-0.106642, 0.984132, -0.141816),
                                                    cameraTarget=(246.817, 205.936, 51.0406), viewOffsetX=-26.3132,
                                                    viewOffsetY=4.29376)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=852.469,
                                                    farPlane=1101.53, width=177.258, height=77.7125,
                                                    cameraPosition=(-740.245,
                                                                    123.051, 523.176),
                                                    cameraUpVector=(-0.159282, 0.976636, -0.144264),
                                                    cameraTarget=(244.537, 200.759, -38.0551), viewOffsetX=-26.3625,
                                                    viewOffsetY=4.3018)
    a = mdb.models[model_name].rootAssembly
    f1 = a.instances['PZT-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#2 ]',), )
    region = a.Set(faces=faces1, name='pzt_1000V')
    mdb.models[model_name].ElectricPotentialBC(name='BC-2', createStepName='Step-1',
                                              region=region, fixed=OFF, distributionType=UNIFORM, fieldName='',
                                              magnitude=500.0, amplitude='five_peaks_120Khz')
    session.viewports['Viewport: 1'].assemblyDisplay.showInstances(instances=(
        'slab-1',))
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF,
                                                               predefinedFields=OFF, interactions=ON, constraints=ON,
                                                               engineeringFeatures=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON,
                                                               predefinedFields=ON, interactions=OFF, constraints=OFF,
                                                               engineeringFeatures=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF,
                                                               bcs=OFF, predefinedFields=OFF, connectors=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models[model_name].parts['slab']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    p = mdb.models[model_name].parts['slab']
    p.seedPart(size=3.0, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models[model_name].parts['slab']
    p.generateMesh()
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD,
                              kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF,
                              hourglassControl=DEFAULT, distortionControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    p = mdb.models[model_name].parts['slab']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    pickedRegions = (cells,)
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2,
                                                       elemType3))
    p = mdb.models[model_name].parts['PZT']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['PZT']
    p.seedPart(size=0.5, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models[model_name].parts['PZT']
    p.generateMesh()
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    elemType1 = mesh.ElemType(elemCode=C3D8E, elemLibrary=STANDARD)
    elemType2 = mesh.ElemType(elemCode=C3D6E, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4E, elemLibrary=STANDARD)
    p = mdb.models[model_name].parts['PZT']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    pickedRegions = (cells,)
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2,
                                                       elemType3))
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    a1 = mdb.models[model_name].rootAssembly
    a1.regenerate()
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON,
                                                               constraints=ON, connectors=ON, engineeringFeatures=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF,
                                                               constraints=OFF, connectors=OFF, engineeringFeatures=OFF)

    mdb.Job(name=Job_name, model=model_name, description='', type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=80,
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
            scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1,
            multiprocessingMode=DEFAULT, numCpus=8, numDomains=8, numGPUs=0)
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
                                                           engineeringFeatures=ON, mesh=OFF)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    p = mdb.models[model_name].parts['PZT']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['slab']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['PZT']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    mdb.models[model_name].sections['PZT'].setValues(material='PZT-5H',
                                                    thickness=None)
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=ON)
    regionDef = mdb.models[model_name].rootAssembly.sets['pzt_1000V']
    mdb.models[model_name].historyOutputRequests['H-Output-1'].setValues(variables=(
        'U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3', 'V1', 'V2', 'V3', 'VR1', 'VR2',
        'VR3'), region=regionDef)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=OFF)
    # mdb.jobs[Job_name].submit(consistencyChecking=OFF)
    #: 作业输入文件 "Job-1.inp" 已提交分析.
    #: Job Job-1: Analysis Input File Processor completed successfully.