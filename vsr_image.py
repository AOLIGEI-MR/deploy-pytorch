# -*- coding: utf-8 -*-
#
#    Filename: vsr_image.py
#
# Description: A partial implementation of the VSR image
#
#      Author: Alexander Fischer, Frank Thiele, 2019-09-19
#
# File format: (see adVsrImageIO.h)
#
#               type            (int)
#               bricksize       (3 * int)
#               name length     (int)
#               name            (char * namelength)
#               physical depth  (unsigned int)
#               logical depth   (unsigned int)
#               maxval          (double)
#               minval          (double)
#               minMaxValidFlag (int)
#               rowlength       (unsigned int)
#               origin          (3 * double)
#               axisX           (3 * double)
#               axisY           (3 * double)
#               axisZ           (3 * double)
#               extent          (3 * double)
#               slices          (bricksize.z)
#               slice 0 - bricksize.z - 1: (image size * sizeof(VoxelType))
#

from __future__ import division
from __future__ import print_function

import itertools
import json
import math
import os
import re
import struct
import sys

import numpy
# global var here only for debugging/testing:
verbosity = 0

class VsrImage(object):
    """
    A class representing a IntelliSpace Discovery VSR image.
    """

    rescaleSlopeTag = '(0028,1053)'
    rescaleInterceptTag = '(0028,1052)'
    realWorldValueSlopeTag = '(0040,9096)[0](0040,9225)'
    realWorldValueInterceptTag = '(0040,9096)[0](0040,9224)'

    def __init__(self, init = None):
        """
        Constructor. Can initialize from file, brickSize or numpy data array.
        """

        self.python3 = sys.version_info >= (3, 0)
        self.name = ''
        self.data = None
        self.origin = numpy.zeros(3, dtype=numpy.float64)
        self.axisX = numpy.eye(3)[0,:]
        self.axisY = numpy.eye(3)[1,:]
        self.axisZ = numpy.eye(3)[2,:]
        self.extent = numpy.zeros(3, dtype=numpy.float64)
        self.metaData = {}
        if type(init) == type('') or type(init) == type(u''):    # filename
            self.read(init)
        elif type(init) == type(()): # data shape
            self.data = numpy.zeros(init, dtype=numpy.float32)
        elif type(init) == type(numpy.zeros(1)): # data array
            self.data = init

    @property
    def voxelSize(self):
        """
        Get the voxel size of the VSR image.
        """

        result = numpy.zeros(3)
        for i in range(3):
            result[i] = self.extent[i] / self.data.shape[i]
        return result

    def crop(self, corner1, corner2):
        """
        Crop the image to a sub box given by the opposite corners.
        """

        if type(corner1) == type([]):
            c1 = numpy.array(corner1).astype(int)
        elif type(corner1) == type(numpy.array(1)):
            c1 = corner1.astype(int)
        if type(corner2) == type([]):
            c2 = numpy.array(corner2).astype(int)
        elif type(corner2) == type(numpy.array(1)):
            c2 = corner2.astype(int)
        self.data = self.data[c1[0]:c2[0],c1[1]:c2[1],c1[2]:c2[2]]

    def get_rescale_slope(self):
        """
        Get RescaleSlope from metadata.
        """
        result = 1.0
        if self.realWorldValueSlopeTag in self.metaData.keys():
            result = self.metaData[self.realWorldValueSlopeTag]
        elif self.rescaleSlopeTag in self.metaData.keys():
            result = self.metaData[self.rescaleSlopeTag]
        return float(result)

    def get_rescale_intercept(self):
        """
        Get RescaleIntercept from metadata.
        """
        result = 0.0
        if self.realWorldValueInterceptTag in self.metaData.keys():
            result = self.metaData[self.realWorldValueInterceptTag]
        elif self.rescaleInterceptTag in self.metaData.keys():
            result = self.metaData[self.rescaleInterceptTag]
        return float(result)

    def set_rescale_slope(self, slope):
        """
        Set RescaleSlope in metadata.
        """
        if self.realWorldValueSlopeTag in self.metaData.keys():
            self.metaData[self.realWorldValueSlopeTag] = '{0}'.format(slope)
        else:
            self.metaData[self.rescaleSlopeTag] = '{0}'.format(slope)

    def set_rescale_intercept(self, intercept):
        """
        Set RescaleIntercept in metadata.
        """
        if self.realWorldValueInterceptTag in self.metaData.keys():
            self.metaData[self.realWorldValueInterceptTag] = '{0}'.format(intercept)
        else:
            self.metaData[self.rescaleInterceptTag] = '{0}'.format(intercept)

    def imageType(self):
        """
        Create the VSR imageType string from the data array dimensions and type.
        """

        typeMap = {}
        typeMap[numpy.dtype('bool')] = 'BINARY'
        typeMap[numpy.dtype('int8')] = 'BYTE'
        typeMap[numpy.dtype('uint8')] = 'BYTE'
        typeMap[numpy.dtype('int16')] = 'GREY'
        typeMap[numpy.dtype('uint16')] = 'GREY'
        typeMap[numpy.dtype('float32')] = 'FLOAT'

        result = 'unknown'
        if self.data.dtype in list(typeMap.keys()):
            result = typeMap[self.data.dtype] + '_{:d}D'.format(len(self.data.shape))
        return result

    def info(self, details=False):
        """
        Print information on the VSR image.
        """

        print('VsrImage', self.imageType())
        print('   name  :', self.name)
        print('    brick:', self.data.shape)
        print('   origin:', self.origin)
        print('    axisX:', self.axisX)
        print('    axisY:', self.axisY)
        print('    axisZ:', self.axisZ)
        print('   extent:', self.extent)
        print('voxelSize:', self.voxelSize)
        if details:
            if self.data.dtype == numpy.bool:
                numActive = numpy.count_nonzero(self.data)
                print('   active:', numActive, 'over slices', self.sliceRange())
            else:
                dataMin, dataMax = self.minmax()
                print('      min:', dataMin)
                print('      max:', dataMax)
            for tag in self.metaData:
                print(tag + ':', self.metaData[tag])

    def matches(self, image):
        """
        Check whether the given image matches with regard to the number of voxels.
        """
        if self.data is None or image.data is None:
            result = False
        if self.data.shape == image.data.shape:
            result = True
        return result

    def minmax(self):
        """
        Computes the minimum and maximum values in one-pass using only 1.5*len(data) comparisons.
        """
        try:
            from itertools import zip_longest as zip_longest  # Python 3
        except:
            from itertools import izip_longest as zip_longest # Python 2

        it = iter(self.data.flatten())
        try:
            lo = hi = next(it)
        except StopIteration:
            raise ValueError('minmax() arg is an empty sequence')
        for x, y in zip_longest(it, it, fillvalue=lo):
            if x > y:
                x, y = y, x
            if x < lo:
                lo = x
            if y > hi:
                hi = y
        return lo, hi

    def read(self, fileName):
        """
        Read VSR image from file.
        """

        # mapping of VSR image type codes to numpy data types
        dataType = {} # see adVsrDefines.h
        dataType[1] = numpy.uint16
        dataType[2] = numpy.bool
        dataType[3] = numpy.float32
        dataType[5] = numpy.uint16
        dataType[6] = numpy.bool
        dataType[7] = numpy.float32
        dataType[11] = numpy.uint8
        dataType[12] = numpy.uint8

        result = None
        with open(fileName, 'rb') as f:
             imageType = struct.unpack('i', f.read(4))[0]
             if not imageType in list(dataType.keys()):
                 print('ERROR: unsupported image type', imageType)
                 return
             brickSize = struct.unpack('3i', f.read(12))
             nameLen = struct.unpack('i', f.read(4))[0]
             self.name = f.read(nameLen).decode("utf-8")
             physicalDepth = struct.unpack('I', f.read(4))[0]
             logicalDepth = struct.unpack('I', f.read(4))[0]
             maxval = struct.unpack('d', f.read(8))[0]
             minval = struct.unpack('d', f.read(8))[0]
             minMaxValidFlag = struct.unpack('I', f.read(4))[0]
             rowLength = struct.unpack('I', f.read(4))[0]
             self.origin = numpy.array(struct.unpack('3d', f.read(24)))
             self.axisX = numpy.array(struct.unpack('3d', f.read(24)))
             self.axisY = numpy.array(struct.unpack('3d', f.read(24)))
             self.axisZ = numpy.array(struct.unpack('3d', f.read(24)))
             self.extent = numpy.array(struct.unpack('3d', f.read(24)))
             dType = dataType[imageType]
             self.data = numpy.zeros(brickSize, dtype=dType)
             if dType == numpy.dtype('bool'):
                 for z in range(brickSize[2]):
                     sliceSize = struct.unpack('i', f.read(4))[0]
                     # convert from uint32 to uint8 taking into account little endian byte ordering
                     sliceData = numpy.frombuffer(f.read(sliceSize), dtype=numpy.uint32, count=int(sliceSize / 4))
                     x = numpy.copy(sliceData).byteswap(True)
                     if self.python3:
                         sliceData = numpy.frombuffer(x.tobytes(), dtype=numpy.uint8, count=sliceSize)
                     else:
                         sliceData = numpy.frombuffer(numpy.getbuffer(x), dtype=numpy.uint8, count=sliceSize)
                     rowSize = int(sliceSize / (brickSize[1] + 2))
                     pos = rowSize
                     for y in range(brickSize[1]):
                         bitData = numpy.unpackbits(sliceData[pos:pos+rowSize])
                         self.data[:,y,z] = bitData[:brickSize[0]]
                         pos += rowSize
             else:
                 sliceSize = int(rowLength * brickSize[1] * physicalDepth / 8)
                 for z in range(brickSize[2]):
                     sliceData = numpy.frombuffer(f.read(sliceSize), dtype=dType, count=brickSize[0] * brickSize[1])
                     self.data[:,:,z] = sliceData.reshape((brickSize[0], brickSize[1]),order='F')

        self.metaData = {}
        jsonFileName = os.path.splitext(fileName)[0] + '.json'
        if not os.path.exists(jsonFileName):
            # json filename might be lowercase (on Unix this really is a different file)
            jsonDirName = os.path.dirname(jsonFileName)
            jsonFileName = os.path.basename(jsonFileName).lower()
            jsonFileName = os.path.join(jsonDirName, jsonFileName)
        if os.path.exists(jsonFileName):
            with open(jsonFileName, 'r') as f:
                self.metaData = json.load(f)
            return
        xmlFileName = os.path.splitext(fileName)[0] + '.xml'
        if not os.path.exists(xmlFileName):
            return
        with open(xmlFileName, 'r') as f:
             for line in f.readlines():
                 m = re.match('\s*<([^>]+)>([^<]+)<[^>]+>', line)
                 if m:
                     self.metaData[m.group(1)] = m.group(2)

    def roll(self, dim, delta):
        """
        Shift the image data along an axis by a specified number of voxels.
        """

        dim2axis = { 'x' : 0, 'y' : 1, 'z' : 2 }
        a = dim2axis[dim.lower()]
        self.data = numpy.roll(self.data, delta, axis=a)

    def show(self, mask=None, factor=1000, colormap='jet'):
        """
        Show the image in a MayaVi based VolumeSlicer.
        """

        from .volume_slicer import VolumeSlicer

        volData = self.data
        if mask != None:
            if type(mask) == type(VsrImage()):
                maskData = mask.data
            else:
                maskData = mask
            volData += maskData.astype(volData.dtype) * factor
        viewer = VolumeSlicer(data=volData, colormap=colormap)
        viewer.configure_traits()

    def sitkImage(self, scaled=False):
        """
        Get SimpleITK image from this VSR image.
        """
        import numpy
        import SimpleITK as sitk
        if self.data.dtype == numpy.dtype('bool'):
            data = numpy.swapaxes(self.data.astype(numpy.uint16), 0, 2)
            result = sitk.GetImageFromArray(data)
        else:
            result = sitk.GetImageFromArray(numpy.swapaxes(self.data, 0, 2))
            if scaled and '(0028,1053)' in self.metaData.keys():
                slope = self.metaData['(0028,1053)']
                intercept = self.metaData['(0028,1052)']
                filter = sitk.CastImageFilter()
                filter.SetOutputPixelType(sitk.sitkFloat32)
                result = filter.Execute(result)
                result = result * slope + intercept
        result.SetOrigin(self.origin)
        result.SetSpacing(self.voxelSize)
        result.SetDirection(numpy.append(numpy.append(self.axisX, self.axisY), self.axisZ))
        return result

    def sliceRange(self):
        """
        Identify the non-zero slice range of the image.
        """
        bottom = 0
        top = self.data.shape[2] - 1
        for z in range(self.data.shape[2]):
            num = numpy.count_nonzero(self.data[:,:,z])
            if num == 0:
                continue
            bottom = z
            break
        if bottom > 0:
            for z in range(self.data.shape[2] - 1, bottom, -1):
                num = numpy.count_nonzero(self.data[:,:,z])
                if num == 0:
                    continue
                top = z
                break
        return bottom, top

    def statistics(self, voi=None):
        """
        Calculate statistics for a given voi on the image.
        """

        result = { 'Num' : 0, 'Vol' : 0.0, 'Min' : 0.0, 'Max' : 0.0, 'Mean' : 0.0, 'StdDev' : 0.0 }
        if voi == None:
            # 1 pass computation according to Knuth
            for value in numpy.nditer(self.data):
                if result['Num'] == 0:
                    result['Min'] = value
                    result['Max'] = value
                else:
                    result['Min'] = min([ result['Min'], value ])
                    result['Max'] = max([ result['Max'], value ])
                d = value - result['Mean']
                result['Mean'] += d / (result['Num'] + 1)
                result['StdDev'] += d * (value - result['Mean'])
                result['Num'] = result['Num'] + 1
        else:
            if not self.matches(voi):
                return None
            mask = numpy.nonzero(voi.data)
            n = len(mask[0])
            for i in range(n):
                x = mask[0][i]
                y = mask[1][i]
                z = mask[2][i]
                value = self.data[x, y, z]
                if result['Num'] == 0:
                    result['Min'] = value
                    result['Max'] = value
                else:
                    result['Min'] = min([ result['Min'], value ])
                    result['Max'] = max([ result['Max'], value ])
                d = value - result['Mean']
                result['Mean'] += d / (result['Num'] + 1)
                result['StdDev'] += d * (value - result['Mean'])
                result['Num'] = result['Num'] + 1
        if result['Num'] > 0:
            result['StdDev'] = math.sqrt(result['StdDev'] / result['Num'])
        result['Vol'] = result['Num'] * numpy.prod(self.voxelSize) / 1000.0
        return result

    def write(self, fileName):
        """
        Write VSR image to file.
        """

        vsrImageType = {} # see adVsrDefines.h
        vsrImageType['GREY_2D'] = 1
        vsrImageType['BINARY_2D'] = 2
        vsrImageType['FLOAT_2D'] = 3
        vsrImageType['GREY_3D'] = 5
        vsrImageType['BINARY_3D'] = 6
        vsrImageType['FLOAT_3D'] = 7
        vsrImageType['BYTE_2D'] = 11
        vsrImageType['BYTE_3D'] = 12

        dir = os.path.abspath(os.path.dirname(fileName))
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(fileName, 'wb') as f:
            imageType = self.imageType()
            f.write(struct.pack('i', vsrImageType[imageType]))
            brickSize = [ self.data.shape[0], self.data.shape[1], self.data.shape[2] ]
            f.write(struct.pack('3i', *brickSize))
            # for unicode support/py3, make sure we have a decent byte string:
            writeName = str(self.name)
            f.write(struct.pack('i', len(writeName)))
            f.write(writeName.encode('utf-8'))
            if imageType == 'BINARY_2D' or imageType == 'BINARY_3D':
                physicalDepth = 1
            else:
                physicalDepth = self.data.dtype.itemsize * 8
            f.write(struct.pack('i', physicalDepth))
            logicalDepth = physicalDepth
            if physicalDepth == 16:
                logicalDepth = 12
            f.write(struct.pack('i', logicalDepth))
            minval, maxval = self.minmax()
            f.write(struct.pack('d', maxval))
            f.write(struct.pack('d', minval))
            f.write(struct.pack('i', 0))
            rowLength = self.data.shape[0]
            f.write(struct.pack('i', rowLength))
            o = numpy.array(self.origin).astype(numpy.float64)
            f.write(struct.pack('3d', *[ o[0], o[1], o[2] ]))
            a = numpy.array(self.axisX).astype(numpy.float64)
            f.write(struct.pack('3d', *[ a[0], a[1], a[2] ]))
            a = numpy.array(self.axisY).astype(numpy.float64)
            f.write(struct.pack('3d', *[ a[0], a[1], a[2] ]))
            a = numpy.array(self.axisZ).astype(numpy.float64)
            f.write(struct.pack('3d', *[ a[0], a[1], a[2] ]))
            e = numpy.array(self.extent).astype(numpy.float64)
            f.write(struct.pack('3d', *[ e[0], e[1], e[2] ]))

            dType = self.data.dtype
            if dType == numpy.dtype('bool'):
                sliceSize = int((brickSize[1] + 2) * (int((brickSize[0] + 31) / 32) + 1))
                sliceSize *= 4 # number of bytes per plane
                rowSize = int(sliceSize / (brickSize[1] + 2))
                for z in range(brickSize[2]):
                    f.write(struct.pack('i', sliceSize))
                    # start with a line of zeros as boundary values
                    sliceData = numpy.zeros(rowSize, dtype=numpy.uint8)
                    for y in range(brickSize[1]):
                        rowData = numpy.packbits((self.data[:,y,z]).view(numpy.uint8))
                        # pad with zeros as trailing boundary values
                        rowData = numpy.pad(rowData, (0, rowSize-len(rowData)), 'constant')
                        sliceData = numpy.append(sliceData, rowData)
                    # add last line of zeros as boundary values
                    sliceData =  numpy.append(sliceData, numpy.zeros(rowSize, dtype=numpy.uint8) )
                    # convert from uint8 to uint32 taking into account little endian byte ordering
                    if self.python3:
                        x = numpy.frombuffer(sliceData.tobytes(), dtype=numpy.uint32, count=int(sliceSize / 4))
                    else:
                        x = numpy.frombuffer(numpy.getbuffer(sliceData), dtype=numpy.uint32, count=int(sliceSize / 4))
                    sliceData = numpy.copy(x).byteswap(True)
                    f.write(sliceData)
            else:
                sliceSize = int(rowLength * self.data.shape[1] * physicalDepth / 8)
                for z in range(self.data.shape[2]):
                    if self.python3:
                        sliceData = numpy.transpose(self.data[:,:,z]).flatten().tobytes()
                    else:
                        sliceData = numpy.getbuffer(numpy.transpose(self.data[:,:,z]).flatten())
                    f.write(sliceData)

        if len(self.metaData) == 0:
            return
        jsonFileName = os.path.splitext(fileName)[0] + '.json'
        with open(jsonFileName, 'w') as f:
            json.dump(self.metaData, f, indent=2, sort_keys=True)

    def write_nifti(self, fileName, scaled=False):
        """
        Write vsr volume to disk in Nifti format.
        NOTE: requires nibabel package
        """
        import nibabel
        outDir = os.path.dirname(fileName)
        if outDir and not os.path.exists(outDir):
            os.makedirs(outDir)
        data = self.data
        description=self.name
        pixdims = numpy.divide(self.extent, data.shape)
        #   in vsr, origin is central position in volume
        #   in nifti and DICOM, origin is center of first voxel
        originVsr = self.origin
        diffCenterOrigin = 0.5 * (self.extent - pixdims)
        xAxis = self.axisX
        yAxis = self.axisY
        zAxis = self.axisZ
        imagePositionPatient = [
            diffCenterOrigin[0] * xAxis[0] + diffCenterOrigin[1] * yAxis[0] + diffCenterOrigin[2] * zAxis[0],
            diffCenterOrigin[0] * xAxis[1] + diffCenterOrigin[1] * yAxis[1] + diffCenterOrigin[2] * zAxis[1],
            diffCenterOrigin[0] * xAxis[2] + diffCenterOrigin[1] * yAxis[2] + diffCenterOrigin[2] * zAxis [2]
        ]
        origin = originVsr - imagePositionPatient
        slope = 1.0
        intercept = 0.0
        if self.metaData != None:
            slope = self.get_rescale_slope()
            intercept = self.get_rescale_intercept()
        if scaled:
            data = data * numpy.float32(slope) + numpy.float32(intercept)
        hdr = self.create_nifti_header(data, origin, self.extent, xAxis, yAxis, description)
        nii = nibabel.Nifti1Image(data, None, hdr)
        hdr = nii.get_header()
        if not scaled:
            # slope and intercept need to be set here AFTER nii creation
            hdr.set_slope_inter(slope, intercept)
        if verbosity: print(hdr)
        nii.to_filename(fileName)
        return nii

    def create_nifti_header(self, data, origin, extent, xAxis=[1.0, 0.0, 0.0], yAxis=[0.0, 1.0, 0.0], desc=None):
        """
        Create Nifti header for given numpy data array.
        xAxis and yAxis vectors refer to vsr/dicom specs, i.e. transposed to nifti matrix
        NOTE: requires nibabel package
        """
        import nibabel

        dataType = { numpy.dtype('bool') : 2, numpy.dtype('uint8') : 2, numpy.dtype('int16') : 4, numpy.dtype('uint16') : 512, numpy.dtype('float32') : 16 }
        voxelSize = [ extent[0] / data.shape[0], extent[1] / data.shape[1], extent[2] / data.shape[2] ]
        hdr = nibabel.Nifti1Header()
        hdr['datatype'] = dataType[data.dtype]
        hdr['dim'] = [ 3, data.shape[0], data.shape[1], data.shape[2], 1, 1, 1, 1 ]
        hdr['pixdim'] = [ 1, voxelSize[0], voxelSize[1], voxelSize[2], 0, 0, 0, 0 ]
        hdr.set_xyzt_units('mm')
        matrix = numpy.eye(4)
        zAxis = numpy.cross(xAxis,yAxis)
        matrix[0,0] = -xAxis[0] * voxelSize[0]
        matrix[1,0] = -xAxis[1] * voxelSize[0]
        matrix[2,0] = xAxis[2] * voxelSize[0]
        matrix[0,1] = -yAxis[0] * voxelSize[1]
        matrix[0,2] = -zAxis[0] * voxelSize[2]

        matrix[1,1] = -yAxis[1] * voxelSize[1]
        matrix[1,2] = -zAxis[1] * voxelSize[2]
        matrix[2,1] = yAxis[2] * voxelSize[1]
        matrix[2,2] = zAxis[2] * voxelSize[2]
        hdr.set_qform(matrix, 2)
        hdr.set_sform(matrix, 1)
        hdr['qoffset_x'] = -origin[0]
        hdr['qoffset_y'] = -origin[1]
        hdr['qoffset_z'] = origin[2]
        hdr['srow_x'] = [ matrix[0,0], matrix[0,1], matrix[0,2], hdr['qoffset_x'] ]
        hdr['srow_y'] = [ matrix[1,0], matrix[1,1], matrix[1,2], hdr['qoffset_y'] ]
        hdr['srow_z'] = [ matrix[2,0], matrix[2,1], matrix[2,2], hdr['qoffset_z'] ]
        hdr['descrip'] = desc if desc is not None else ''
        return hdr

    # convert nifti to vsr
    # if voiFile==True, pixel array is tried to convert to bool,
    #    and then written as binary vsr
    # if voiFile==False and convertTo16Bit==True, pixel array is converted to 16bit
    def read_nifti (self, nii, voiFile=False, threshold=0.5, convertTo16Bit=False):
        import nibabel as nib

        # read nii file; nib can also handle .gz files
        if isinstance(nii, nib.Nifti1Image):
            imgNii = nii
            niiName = "Nifti Image"
        else:
            imgNii = nib.load(nii)
            niiName = nii

        imgData = imgNii.get_data()
        imgHdr  = imgNii.get_header()
        if verbosity>1: print (imgHdr)

        # parse nifti header
        dims = imgHdr.get_data_shape()
        pixdims = imgHdr.get_zooms()
        slope, inter = imgHdr.get_slope_inter()
        unitSpace, unitTime = imgHdr.get_xyzt_units()
        dataType = imgHdr.get_data_dtype()

        # check if image should be converted to voi vsr:
        if voiFile:
            imgDataSpectrum = numpy.unique(imgData.flatten())
            minVal = imgDataSpectrum[0]
            if len(imgDataSpectrum) > 2:
                print ("WARNING: more than 2 values in image data: ", imgDataSpectrum)
                print ("         values > threshold (%g) will be mapped to 1" %threshold)
            # shift minVal to zero:
            imgData = imgData - minVal
            # set everything else to 1
            maskIdx = imgData > threshold
            imgData[maskIdx]=1
            # and convert to bool
            imgData = numpy.bool8(imgData)
            dataType = numpy.bool8
        elif convertTo16Bit:
            if verbosity: print('converting data to uint16')
            dataType = numpy.uint16

        # check compatibility of nifti image:
        if unitSpace != 'mm':
            print ("WARNING: spacial unit %s not implemented yet, treating as mm" % unitSpace )
        if ( (slope != None and slope != 1.0) or (inter!=None and inter != 0.0) ):
            print ("WARNING: found slope/intercept - will be ignored!")
            print ("         (scaling needs to be implemented if needed)")
        if dataType == numpy.int16 or dataType == numpy.uint16:
            print ("creating GREY_3D vsr.")
            imgData = numpy.uint16(imgData)
        elif dataType == numpy.int8 or dataType == numpy.uint8:
            print ("creating BYTE_3D vsr.")
            imgData = numpy.uint8(imgData)
        elif dataType == numpy.float32:
            print ("creating FLOAT_3D vsr.")
            imgData = numpy.float32(imgData)
        elif dataType == numpy.bool8:
            print ("creating BINARY_3D vsr.")
            imgData = numpy.bool8(imgData)
        else:
            print ("dtype %s not implemented yet, skipping" % dataType)
            return

        # transform coordinate system (RAS to LPS)
        axisX,axisY,axisZ,origin,flipAP = self.nii2dicomCoords( imgHdr )
        #  in case of left-handed data, flip 2nd coordinate
        if flipAP:
            imgData=imgData[:,::-1,:]
        if verbosity>1:
            print ("Vsr axisX = %.4f, %.4f, %.4f" %(axisX[0],axisX[1],axisX[2] ))
            print ("Vsr axisY = %.4f, %.4f, %.4f" %(axisY[0],axisY[1],axisY[2] ))
            print ("Vsr axisZ = %.4f, %.4f, %.4f" %(axisZ[0],axisZ[1],axisZ[2] ))
            print ("Vsr origin = %.4f, %.4f, %.4f" %(origin[0],origin[1],origin[2] ))

        # set vsr image properties
        self.name = os.path.basename(niiName)
        self.brickSize = dims
        self.extent = list(map(lambda x,y:x*y,dims,pixdims))
        self.origin = origin
        self.axisX = axisX
        self.axisY = axisY
        self.axisZ = axisZ
        self.data = imgData

        # add slope and intercept to metadata
        if slope != None and inter != None and (slope != 1.0 or inter != 0.0):
            self.metaData['(0028,1052)'] = '%f'%inter
            self.metaData['(0028,1053)'] = '%f'%slope

        # for testing
        if verbosity>1:
            self.info(details=True)
        return 0

    # convert nifti RAS coordinate system to dicom (LPS)
    def nii2dicomCoords(self, niiHdr ):
        # get nifti matrix
        # follow nifti standard
        matrix = numpy.asarray( niiHdr.get_best_affine() )
        dims =  numpy.asarray( niiHdr.get_data_shape() )
        pixdims =  numpy.asarray( niiHdr.get_zooms() )
        extents =  dims * pixdims
        diffCenterOrigin = 0.5*(extents - pixdims)

        # 0. check nifti geometry
        if verbosity>1: print (matrix)
        flipAP = False
        if numpy.linalg.det( matrix ) <0:
            print ("nifti matrix has negative determinant: %g"% numpy.linalg.det( matrix ) )
            print ("Trying to fix by flipping voxel array in 2nd dimension.")
            print ("Warning: the fix only works exactly for diagonal matrices!")
            flipAP = True
            # flip j coord for x,y,z
            matrix[1][:] = -matrix[1][:]
            # and shift adapt origin by 1 pixeldim
            # only tested for diag matrix
            matrix[1][3] = matrix[1][3] + pixdims[1]

        # 1. transform from DICOM to nifti coordinate system:
        # nifti should have (right-handed) RAS ordering
        # DICOM patient co-ordinate system: right-handed LPS: x increases R -> L, y increases A -> P, z increases I -> S
        # HERE, only adapt matrix to compute x,y,z in LPS, without transforming dicom voxel storage

        # flip x and y coordinates: ras_to_lps = diag([-1.0 -1.0 1.0 1.0]);
        vsrX=[-matrix[0][0]/pixdims[0],-matrix[1][0]/pixdims[0], matrix[2][0]/pixdims[0]]
        vsrY=[-matrix[0][1]/pixdims[1],-matrix[1][1]/pixdims[1], matrix[2][1]/pixdims[1]]
        vsrZ=[-matrix[0][2]/pixdims[2],-matrix[1][2]/pixdims[2], matrix[2][2]/pixdims[2]]

        # adapt origin:
        # - in nifti and DICOM, origin is center of first voxel
        firstVoxelCenter = numpy.asarray([ -matrix[0,3],-matrix[1,3], matrix[2,3] ])
        imagePositionPatient = numpy.asarray([ diffCenterOrigin[0]*vsrX[0] + diffCenterOrigin[1]*vsrY[0] + diffCenterOrigin[2]*vsrZ[0],
                                  diffCenterOrigin[0]*vsrX[1] + diffCenterOrigin[1]*vsrY[1] + diffCenterOrigin[2]*vsrZ[1],
                                    diffCenterOrigin[0]*vsrX[2] + diffCenterOrigin[1]*vsrY[2] + diffCenterOrigin[2]*vsrZ[2] ])
        # - in vsr, origin is central position in volume
        origin = firstVoxelCenter + imagePositionPatient
        return vsrX,vsrY,vsrZ,origin,flipAP
