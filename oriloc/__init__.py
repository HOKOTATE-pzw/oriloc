'''
本库用于在一定条件下进行定位
在使用中，距离的单位保持为km

This package is used for location under certain conditions
The unit of distance is maintained as "km"
'''
import numpy as np


def _construct_equation_matrix(c, d, theta, n):
    A = np.array([[2 * (c[1, 0] - c[0, 0]), 2 * (c[1, 1] - c[0, 1])],
                  [np.sin(theta[n]), -np.cos(theta[n])]])
    C = np.array([[d[0]**2 - d[1]**2 + c[1, 0]**2 - c[0, 0]**2 + c[1, 1]**2 - c[0, 1]**2],
                  [np.sin(theta[n]) * c[0, 0] - np.cos(theta[n]) * c[0, 1]]])
    return A, C


def distance(c1, c2, rad=False):
    '''
    求已知两点之间的距离
    
    参数
    ----
    c1：一维数组，第一点的经纬度
    c2：一维数组，第二点的经纬度
    rad：布尔类型，可选，True表示输入经纬度为弧度制，默认为False
    
    
    Find the distance between two known points
    
    Parameters
    ----------
    c1: 1D array, the longitude and the latitude of the first point
    c2: 1D array, the longitude and the latitude of the second point
    rad: bool, callable, True indicates that the input longitude and latitude are in radian system, default=False
    '''
    c1 = np.array(c1)
    c2 = np.array(c2)
    if rad is False:
        c1 *= np.pi / 180
        c2 *= np.pi / 180
    
    if len(c1.shape) == 1 and len(c2.shape) == 1:
        return 6371.393 * np.arccos(np.sin(c1[1]) * np.sin(c2[1]) + np.cos(c1[1]) * np.cos(c2[1]) * np.cos(c1[0] - c1[1]))
    elif len(c1.shape) == 1 and len(c2.shape) == 2:
        return 6371.393 * np.arccos(np.sin(c1[1]) * np.sin(c2[:, 1]) + np.cos(c1[1]) * np.cos(c2[:, 1]) * np.cos(c1[0] - c2[:, 0]))
    elif len(c1.shape) == 2 and len(c2.shape) == 1:
        return 6371.393 * np.arccos(np.sin(c1[:, 1]) * np.sin(c2[1]) + np.cos(c1[:, 1]) * np.cos(c2[1]) * np.cos(c1[:, 0] - c2[0]))
    else:
        return 6371.393 * np.arccos(np.sin(c1[:, 1]) * np.sin(c2[:, 1]) + np.cos(c1[:, 1]) * np.cos(c2[:, 1]) * np.cos(c1[:, 0] - c2[0]))


def tria(c, d):
    '''
    三角定位法
    
    参数
    ----
    c：二维数组，已知n点的经纬度，n>=3
        [[lon1, lat1],
         [lon2, lat2],
         [lon3, lat3],
         ...]
    d：一维数组，n点距离目标点的距离[d1, d2, d3, ...]
    
    
    Triangulation method
    
    Parameters
    ----------
    c: 2D array, known longitude and latitude of n points, n>=3
        [[lon1, lat1],
         [lon2, lat2],
         [lon3, lat3],
         ...]
    d: 1D array, the distances between n points and the destination[d1, d2, d3, ...]
    '''
    c = np.array(c)
    d = np.array(d)
    c *= np.pi / 180
    lat_range = []
    for i in range(d.shape[0]):
        lat_range.append((d[i] + 0.05) / 6371.393)
    lat_range = np.array(lat_range)
    lat_range = [(c[:, 1] - lat_range).min(), (c[:, 1] + lat_range).max()]
    
    for i in range(2):
        c_copy = c.copy()
        if lat_range[1] != lat_range[0]:
            r2 = lat_range[1] - lat_range[0]
            r2 += np.sin(lat_range[1]) * np.cos(lat_range[1]) - np.sin(lat_range[0]) * np.cos(lat_range[0])
            r2 /= np.sin(lat_range[1]) - np.sin(lat_range[0])
            r2 *= 3185.6965
        else:
            r2 = 6371.393 * np.cos(lat_range[0])
        c_copy[:, 0] *= r2
        c_copy[:, 1] *= 6371.393
        
        for j in range(c.shape[0]-1):
            for k in range(j+1, c.shape[0]):
                if k == 1:
                    A = np.array([c_copy[k,0]-c_copy[j,0], c_copy[k,1]-c_copy[j,1]])
                    Y = np.array([d[j]**2 - d[k]**2 + c_copy[k,0]**2-c_copy[j,0]**2 + c_copy[k,1]**2 - c_copy[j,1]**2])
                else:
                    A = np.vstack((A, np.array([c_copy[k,0]-c_copy[j,0], c_copy[k,1]-c_copy[j,1]])))
                    Y = np.vstack((Y,
                                   np.array([d[j]**2 - d[k]**2 + c_copy[k,0]**2-c_copy[j,0]**2 + c_copy[k,1]**2 - c_copy[j,1]**2])))
        
        x = np.linalg.lstsq(2 * A, Y, rcond=None)[0].T[0]
        x[0] /= r2
        x[1] /= 6371.393
        
        if i == 0:
            # 进一步缩小纬度范围
            if x[1] <= c[:, 1].max() and x[1] >= c[:, 1].min():
                lat_range = [c[:, 1].min(), c[:, 1].max()]
            elif x[1] < c[:, 1].min():
                lat_range = [x[1] - 0.01 * d.mean() / 6371.393, c[:, 1].max()]
            elif x[1] > c[:, 1].max():
                lat_range = [c[:, 1].min(), x[1] + 0.01 * d.mean() / 6371.393]
    
    return x * 180 / np.pi


def cir(c, d):
    '''
    地球上的两点做圆求交点，三角定位法的补充
    
    参数
    ----
    c：二维数组，已知两点的经纬度
        [[lon1, lat1],
         [lon2, lat2]]
    d：一维数组，已知两点点距离目标点的距离[d1, d2]
    
    
    Two points on the earth are rounded to find the intersection point
    which is a supplement to the triangulation method
    
    Parameters
    ----------
    c: 2D array, known longitude and latitude of two points
        [[lon1, lat1],
         [lon2, lat2]]
    d: 1D array, the distances between the known two points and the destination[d1, d2]
    '''
    c = np.array(c)
    d = np.array(d)
    c *= np.pi / 180
    lat_range = []
    for i in range(d.shape[0]):
        lat_range.append((d[i] + 0.05) / 6371.393)
    lat_range = np.array(lat_range)
    lat_range = [[(c[:, 1] - lat_range).min(), (c[:, 1] + lat_range).max()]]
    
    result = []
    for i in range(3):
        c_copy = c.copy()
        if lat_range[i][1] != lat_range[i][0]:
            r2 = lat_range[i][1] - lat_range[i][0]
            r2 += np.sin(lat_range[i][1]) * np.cos(lat_range[i][1]) - np.sin(lat_range[i][0]) * np.cos(lat_range[i][0])
            r2 /= np.sin(lat_range[i][1]) - np.sin(lat_range[i][0])
            r2 *= 3185.6965
        else:
            r2 = 6371.393 * np.cos(lat_range[i][0])
        c_copy[:, 0] *= r2
        c_copy[:, 1] *= 6371.393
        
        # 求圆心连线所在直线的倾斜角
        if c_copy[1, 0] != c_copy[0, 0]:
            theta = np.arctan((c_copy[1, 1] - c_copy[0, 1]) / (c_copy[1, 0] - c_copy[0, 0]))
            
        else:
            theta = np.pi / 2
        
        # pr为圆心到交点连线的距离
        pr = 2 * (c_copy[1, 0] - c_copy[0, 0]) * c_copy[0, 0] + 2 * (c_copy[1, 1] - c_copy[0, 1]) * c_copy[0, 1]
        pr += d[1]**2 - d[0]**2 + c_copy[0, 0]**2 - c_copy[1, 0]**2 + c_copy[0, 1]**2 - c_copy[1, 1]**2
        pr = 0.5 * abs(pr) / (((c_copy[1, 0] - c_copy[0, 0])**2 + (c_copy[1, 1] - c_copy[0, 1])**2))**0.5
        alpha = np.arccos(pr / d[0])
        theta = (theta + alpha, theta - alpha)
        
        if i == 0:
            # 初步定位两个点的位置
            A, C = _construct_equation_matrix(c_copy, d, theta, 0)
            xs = np.linalg.solve(A,C).T[0]
            
            A, C = _construct_equation_matrix(c_copy, d, theta, 1)
            xs = np.vstack((xs, np.linalg.solve(A,C).T[0]))
            
            xs[:, 0] /= r2
            xs[:, 1] /= 6371.393
            # 进一步缩小纬度范围
            for x in xs:
                if x[1] <= c[:, 1].max() and x[1] >= c[:, 1].min():
                    lat_range.append([c[:, 1].min(), c[:, 1].max()])
                elif x[1] < c[:, 1].min():
                    lat_range.append([x[1] - 0.01 * d.mean() / 6371.393, c[:, 1].max()])
                elif x[1] > c[:, 1].max():
                    lat_range.append([c[:, 1].min(), x[1] + 0.01 * d.mean() / 6371.393])
        
        else:
            # 分别细化两个点的坐标
            A, C = _construct_equation_matrix(c_copy, d, theta, i-1)
            x = np.linalg.solve(A,C).T[0]
            x[0] /= r2
            x[1] /= 6371.393
            result.append(x)
    
    return np.array(result) * 180 / np.pi


def angle(c, theta, base=None):
    '''
    已知两点和两点到目标点的角度，求目标点
    
    参数
    ----
    c：二维数组，已知两点的经纬度
        [[lon1, lat1],
         [lon2, lat2]]
    theta：一维数组，已知两点到目标点的角度[theta1, theta2]
    base：字符串类型或列表类型，可选，基准方位，默认为东
    
    
    Given two points and the angle from two points to the target point, find the target point
    
    Parameters
    ----------
    c：二维数组，已知两点的经纬度
        [[lon1, lat1],
         [lon2, lat2]]
    theta: 1D array, the angles between the known two points and the destination[theta1, theta2]
    base: str or list, callable, base orientation, default to East
    '''
    c = np.array(c) * np.pi / 180
    theta = np.array(theta) * np.pi / 180
    lat_range = np.array([c[:, 1].min(), c[:, 1].max()])
    
    if type(base) == str:
        if base == 'n' or base == 'N':
            theta += 90
        elif base == 'w' or base == 'W':
            theta += 180
        elif base == 's' or base == 'S':
            theta += 270
    
    elif type(base) == list:
        for i in range(len(base)):
            if base[i] == 'n' or base[i] == 'N':
                theta[i] += 90
            elif base[i] == 'w' or base[i] == 'W':
                theta[i] += 180
            elif base[i] == 's' or base[i] == 'S':
                theta[i] += 270
    
    for i in range(2):
        c_copy = c.copy()
        if lat_range[1] != lat_range[0]:
            r2 = lat_range[1] - lat_range[0]
            r2 += np.sin(lat_range[1]) * np.cos(lat_range[1]) - np.sin(lat_range[0]) * np.cos(lat_range[0])
            r2 /= np.sin(lat_range[1]) - np.sin(lat_range[0])
            r2 *= 3185.6965
        else:
            r2 = 6371.393 * np.cos(lat_range[0])
        c_copy[:, 0] *= r2
        c_copy[:, 1] *= 6371.393
        
        A = np.array([[np.sin(theta[0]), -np.cos(theta[0])],
                      [np.sin(theta[1]), -np.cos(theta[1])]])
        C = np.array([[np.sin(theta[0]) * c_copy[0, 0] - np.cos(theta[0]) * c_copy[0, 1]],
                      [np.sin(theta[1]) * c_copy[1, 0] - np.cos(theta[1]) * c_copy[1, 1]]])
        x = np.linalg.solve(A,C).T[0]
        x[0] /= r2
        x[1] /= 6371.393
    
        if i == 0:
            # 进一步缩小纬度范围
            if x[1] <= c[:, 1].max() and x[1] >= c[:, 1].min():
                return x * 180 / np.pi
            else:
                d = np.array([distance(c[0], x, True), distance(c[1], x, True)])
                if x[1] < c[:, 1].min():
                    lat_range = [x[1] - 0.01 * d.mean() / 6371.393, c[:, 1].max()]
                elif x[1] > c[:, 1].max():
                    lat_range = [c[:, 1].min(), x[1] + 0.01 * d.mean() / 6371.393]
    return x * 180 / np.pi