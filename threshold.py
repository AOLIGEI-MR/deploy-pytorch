#
# Filename: threshold.py
#
# Description: thresholding on an input image to create a mask
#

import logging
import os
import platform
import shutil
import stat
import sys

import nibabel
import numpy

from vsr_image import VsrImage

def log_vsr_image_details(vsr, with_minmax=False):
    """
    Log details of a VSR image.
    """
    log.info('VsrImage {0}'.format(vsr.imageType()))
    log.info('     name: {0}'.format(vsr.name))
    log.info('    brick: {0}'.format(vsr.data.shape))
    log.info('   origin: {0}'.format(vsr.origin))
    log.info('    axisX: {0}'.format(vsr.axisX))
    log.info('    axisY: {0}'.format(vsr.axisY))
    log.info('    axisZ: {0}'.format(vsr.axisZ))
    log.info('   extent: {0}'.format(vsr.extent))
    log.info('voxelSize: {0}'.format(vsr.voxelSize))
    if 'BINARY' in vsr.imageType():
        return
    if with_minmax:
        mm = vsr.minmax()
        log.info('   minmax: {0}/{1}'.format(mm[0], mm[1]))
    intercept = vsr.get_rescale_intercept()
    slope = vsr.get_rescale_slope()
    log.info('intercept: {0}'.format(intercept))
    log.info('    slope: {0}'.format(slope))

def log_nifti_image_details(file_name):
    """
    Log details of a Nifti image
    """
    log.debug('NIFTI IMAGE {0}'.format(file_name))
    nii = nibabel.load(file_name)
    log.debug(nii.get_header())
    return nii

# job administration parameters
scriptFileName = sys.argv[0]
scriptDir = os.path.dirname(scriptFileName)
statusFileName = sys.argv[-1]
logFileName = sys.argv[-2]
jobDebug = False

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=logFileName
    )
log = logging.getLogger()
log.setLevel(logging.INFO)

scriptDir = os.path.dirname(os.path.abspath(scriptFileName))
# output parameter directory
outputParamsDir = sys.argv[-3]
jobDir = os.path.dirname(statusFileName)
os.chdir(jobDir)
log.info('starting in {0}'.format(scriptDir))
log.info('LogFileName = {0}'.format(logFileName))
log.info('StatusFileName = {0}'.format(statusFileName))
log.info('OutputParamsDir = {0}'.format(outputParamsDir))
log.info('changing to {0}'.format(jobDir))

#
# input parameters
#
inputImageDir = sys.argv[1]
inputImage = os.path.basename(inputImageDir)
log.info('{0} = {1}'.format(inputImage, inputImageDir))
inputMaskDir = sys.argv[2]
inputMaskName = os.path.basename(inputMaskDir)
log.info('{0} = {1}'.format(inputMaskName, inputMaskDir))
lowerThreshold = float(sys.argv[3])
log.info('{0} = {1}'.format('Lower Threshold', lowerThreshold))
upperThreshold = float(sys.argv[4])
log.info('{0} = {1}'.format('Upper Threshold', upperThreshold))

inputImage = VsrImage(os.path.join(inputImageDir, inputImage + '.vsr'))
log.info('INPUT IMAGE')
log_vsr_image_details(inputImage)
# Nifti is not actually used here, but the export is already shown
# as a lot of other packages work on this format
inputImage.write_nifti('input.nii', scaled=True)
log_nifti_image_details('input.nii')

inputMask = None
if inputMaskDir.endswith('__NONE__'):
    log.warning('no mask')
else:
    inputMask = VsrImage(os.path.join(inputMaskDir, inputMaskName + '.vsr'))
    log.info('MASK')
    log_vsr_image_details(inputMask)

#
# core calculation
#
slope = inputImage.get_rescale_slope()
intercept = inputImage.get_rescale_intercept()
thresholdData = numpy.zeros(inputImage.data.shape, dtype=numpy.bool)
for z in range(inputImage.data.shape[2]):
    log.info('thresholding slice {0}'.format(z))
    for y in range(inputImage.data.shape[1]):
        for x in range(inputImage.data.shape[0]):
            if inputMask and not inputMask.data[x,y,z]:
                continue
            value = inputImage.data[x,y,z] * slope + intercept
            if value >= lowerThreshold and value < upperThreshold:
                thresholdData[x,y,z] = True

statusMessage = 'thresholded {0}-{1}'.format(lowerThreshold, upperThreshold)

#
# output parameters
#
# names must match the definition in the corresponding config.json of the computation node
# they have to be directly placed directly in the output parameter directory (no subdirectory)!
#
segmentationParName = "Segmentation"
statusMessageParName = "Status"
dirMode = stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH
fileMode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IWGRP

# output Segmentation
vsrMask = VsrImage(thresholdData)
vsrMask.origin = inputImage.origin
vsrMask.axisX = inputImage.axisX
vsrMask.axisY = inputImage.axisY
vsrMask.axisZ = inputImage.axisZ
vsrMask.extent = inputImage.extent
vsrMask.name = segmentationParName
outputSegmentation = os.path.join(outputParamsDir, segmentationParName)
log.debug('{0} = {1}'.format(segmentationParName, outputSegmentation))
log.info('OUTPUT {0}'.format(segmentationParName))
log_vsr_image_details(vsrMask)
vsrMask.write(outputSegmentation + '.vsr')
os.chmod(outputSegmentation + '.vsr', fileMode)

# output Status message
statusMessageParDir = os.path.join(outputParamsDir, statusMessageParName)
if not os.path.exists(statusMessageParDir):
    os.makedirs(statusMessageParDir)
os.chmod(statusMessageParDir, dirMode)
statusMessageFile = os.path.join(statusMessageParDir, statusMessageParName + '.txt')
with open(statusMessageFile, 'w') as f:
    f.write(statusMessage)
os.chmod(statusMessageFile, fileMode)

# job status
with open(statusFileName, 'w') as f:
    f.write(statusMessage + '\n')
log.info(statusMessage)

if jobDebug:
    #
    # pack the complete job directory because it gets automatically deleted
    # by the infrastructure no matter whether the job fails or succeeds
    #
    log.debug('backing up job directory {0}'.format(jobDir))
    dirName = os.path.dirname(jobDir)
    os.chdir(dirName)
    fileName = os.path.basename(jobDir)
    if platform.platform().startswith('Windows'):
        cmd = '"C:\\Program Files\\7-Zip\\7z.exe" a -r {0}.zip {1}/'.format(fileName, fileName)
    else:
        cmd = 'tar -czf {0}.tar.gz {1}/'.format(fileName, fileName)
    os.system(cmd)

log.info('finished in {0}'.format(scriptDir))
