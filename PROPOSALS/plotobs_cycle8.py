"""
    Version: 1.1

    Author
       John Carpenter (jcarpent @ alma.cl)

    Change history:
       September 13, 2019 : First release of script (JMC)
       March 10, 2020 : Adapted to Cycle 8 and minor tweaks in description (G.Marinello, S.Martin)

    Purpose
       plotobs_cycle8.py prints and plots the pointing positions of active
       Cycle 7 Grade A projects and ACA standalone A+B projects. The ALMA Archive 
       should be consulted for the list of completed observations over all 
       Cycles.

       The script provides three basic function:
          1) Plot active observations around a specified source name or coordinate,
             and overlay a proposed observation (single pointing or 
             rectangular mosaic) on the plot.

          2) List the sources observed for an active project

          3) List the details (sensitivity, angular resolution, etc...)
             of active observations

    Support
       plotobs_cycle8.py is user-contributed software that is distributed 
       "as-is". plotobs_cycle8.py is not an offical ALMA product and is 
       not supported by the ALMA Regional Centers (ARCs). Please report any 
       bugs to John Carpenter.

    Dependencies
       1) Theplotobs_cycle8.py script requires a data file containing
          the list of ALMA active observations. The root name of the file is 
          contained in the variable LIST_OF_OBSERVATIONS, and the data file
          can be found on the ALMA Science Portal in the link 
          http://almascience.org/proposing/duplications.

          The science portal has two versions of the data: an excel spreadsheet
          (.xlsx extension) and a comma-separated-variable (.csv extension) 
          format. Both formats work with this script. To read the excel
          file, the python package "xlrd" must be installed. My experience
          is that this package is often not installed, and using the .csv 
          file is more likely to be successful.

       2) Various python packages must be installed, which are listed below 
          under "Required software".

    To run the program
       # Start python with graphics enabled by the --pylab option
       ipython --pylab

       # Import program
       import plotobs_cycle8 as po

       # Main routines
       po.plot(...)     # Plots ALMA observations and a proposed observation
       po.project(...)  # Show observations for ALMA projects
       po.row(...)      # List observing parameters for rows in spreadsheet

    Examples
       # Plot scheduled observations within 100" of NGC6300 along 
       # with a proposed pointing at the position of the galaxy
       po.plot('NGC6300', plotsize=200)

       # Plot scheduled observations within 600" x 600" of NGC7496, along with a 
       # proposed mosaic at 345 GHz
       po.plot('NGC6300', plotsize=600, length=240, width=120, 
               pa=60, freq=345., mosaic=True)

       # Plot scheduled observations within 480" of (ra, dec) = (17h16m59.5s, -62d49m14.0s)
       po.plot('17h16m59.5s', '-62d49m14.0s', plotsize=480, frame='icrs', unit='deg')

       # Plot scheduled observations in a 1 deg region around ra, dec = (285.2,-36.9) J2000
       po.plot(285.2,-36.9, plotsize=3600)

       # Plot scheduled observations in a 20 arcmin around galactic center, but mosaics only
       po.plot(0, 0, frame='galactic', plotsize=1200., mosonly=True)

       # Plot scheduled observations within a 120" x 120" of HL Tau and GO Tau
       po.plot('W_Hya, V1094Sco', plotsize=120)

       # Plot scheduled observations near (default to 120") coordinates in row 1448 of the spreadsheet
       po.plot(1448)
       
       # Plot scheduled observations within 1200" centered on the coordinates in row 1448 of the spreadsheet
       po.plot(1588,plotsize=1200)

       # Plot remaining observations centered on row 1588 in the spreadsheet, and only row 1588
       po.plot(1448, include=1448)

       # Plot remaining observations for sources listed in a text file 
       # Contents of input file "input.txt":
W_Hya
V1094Sco
plotsize = 480
NGC6300
frame = galactic
plotsize = 120
236.8775 42.7224
       po.plot(inputFile='input.txt')

       # List scheduled observations for project 2019.1.00195.L
       po.project('2019.1.00195.L')

       # Print the observational parameters for row 130 in Excel spreadsheet
       po.row(130)

    Notes
       1) After running "import plotobs as po", additional help can 
          be obtained by typing in python:
              help(po)
              help(po.plot)
              help(po.row)
              help(po.project)

       2) When plotobs_cycle8.py is run the first time in a python session, the 
          spreadsheet containing the observations is read into memory and 
          stored in the global variable OBSERVATIONS. The data stored in 
          memory is then used on subsequent calls. 

       3) Target of Opportunity or solar system observations will likely (but 
          not always) have a RA/Dec coordinate of (0 deg, 0 deg). Such 
          observations can be identified by running the script as:
             po.plot(0, 0)

    Cautions
       1) The spreadsheet used by plotobs_cycle8.py contains the sensitivity
          and angular resolution requested by the principal investigator. 
          The achieved sensitivity and angular resolution of the actual 
          observations may differ.

       2) The sensitivity per spectral window is computed using the reference
          frequency and reference bandwidth, but does not take into account
          the variation in the system temperature amongst the various
          spectral windows.

    Software
       Platforms
           1) plotobs_cycle8.py program has been successfully tested on Linux,
           MacBook Pro, and an iMac running OS Sierra.

       The program has successfully worked with the following:
           1) python 3.5
           2) numpy 1.15.4
           3) pandas version 0.23.4; see http://pandas.pydata.org .
           4) astropy version 2.0; see http://www.astropy.org
           5) If you want to read the excel (.xlsx) instead of the csv version 
              of the spreadsheet, the xlrd python package needs to be installed.

       Internet connection
           plotobs_cycle8.py does not require a internet connection to run.
           plotobs_cycle8.py does use the Sesame database to resolve the 
           coordinates for source names. If an internet connection is 
           not available, the script cannot use Sesame, and will to determine 
           the coordinates by looking for the source name in the spreadsheet.
"""

# Load packages
import matplotlib.pylab as py
import numpy as np
import pandas as pd
import re
import os
import astropy
from astropy.coordinates import SkyCoord
import astropy.constants as G
from matplotlib.patches import Circle, Polygon

# Seaborn package is not required, but it makes the plot looks nicer
try:
  import seaborn as sns
except:
  pass

# Name of columns in observations
ANG_RES         = 'Req. Ang. Res.'
BAND            = 'Band'
DEC_DEG         = 'Dec'
DEC_DEG_ORIG    = 'Dec_orig'
DEC_DMS         = 'Dec_DMS'
EXCEL_LINE      = 'excel'          # Created in readObservations
FWHM_PB         = 'FWHM PB'        # Created in readObservations
IS_ACA_STANDALONE = 'standAlone_ACA'
IS_LP           = 'Is Large Program' # Created in readObservations
IS_MOSAIC       = 'Mosaic'
IS_SOLAR        = 'Solar Observing'
IS_SPECTRAL     = 'isSpectralScan' # Created in readObservations
IS_SPW_SKY_FREQ = 'Is Sky Freq?'
IS_VLBI         = 'VLBI'
LAS             = 'Req. LAS'
LAT_OFFSET      = 'Lat Offset'
LON_OFFSET      = 'Long Offset'
MOS_AREA        = 'mosaic area'  # Created in readObservations
MOS_LENGTH      = 'Mos. Length'
MOS_PA          = 'Mos. PA'
MOS_WIDTH       = 'Mos. Width'
MAX_SIZE        = 'Max size'     # Created in readObservations
MOS_SPACING     = 'Mos. Spacing'
MOS_COORD       = 'Mos. Coord.'
MOVING_OBJECT   = 'Moving object'   # Set in readObservation
PINAME          = 'PI'
POLARIZATION    = 'Polarization'
PROJECT         = 'Project Code'
RA_DEG          = 'RA'
RA_DEG_ORIG     = 'RA_orig'
RA_HMS          = 'RA_HMS'
REF_FREQ        = 'Ref.Frequency'
REF_BW          = 'Ref.Freq.Width'
REF_BW_MEASURE  = 'repBandWidth_Measure'
REF_BW_UNIT     = 'repBandwidthSG_unit'
REP_WIN_BW      = 'bandwidth_SPW_rep'
REP_WIN_RES     = 'spectralRes_SPW_rep'
REQ_SENS        = 'Req.Sensitivity'
REQ_SENS_UNIT   = 'sensitivity_unit'
SG_FULLNAME     = 'Science Goal'
SG_NAME         = 'SG_Number'
SPS_BW          = 'SPS Bandwidth'
SPS_END         = 'SPS End Freq.'
SPS_SPW_RES     = 'SPS Spec. Res.'
SPS_START       = 'SPS Start Freq.'
SPW_FREQ        = {}
SPW_BW          = {}
SPW_RES         = {}
TARGET          = 'Target Name'
TARGET_STRIPPED = 'Target Name stripped' # Add by readObservations()
TIME_TOTAL      = 'estimatedTime'
TIME_12M        = 'est12Time'
TIME_ACA        = 'estACATime'
TIME_7M         = 'est7Time'
TIME_TP         = 'eTPTime'
USE_12M         = 'Use 12m?'  # Added by readObservations()
USE_7M          = 'Use 7-m?'
USE_TPA         = 'Use TP?'
VELOCITY        = 'Velocity'
VELOCITY_UNIT   = 'Velocity unit'   # added by readObservations()
VELOCITY_REF    = 'Vel. Frame'
VELOCITY_DOP    = 'Vel. Convention'

# Set array keywords
NWINDOWS = 16
for win in range(1, NWINDOWS+1):
   SPW_FREQ[win]        = 'Freq SPW %d' % win
   SPW_BW[win]          = 'Bandwidth SPW %d' % win
   SPW_RES[win]         = 'Spec.Res. SPW %d' % win

# List of Cycle 1/2/3 observations
LIST_OF_OBSERVATIONS = 'duplications_cycle8_march12'
SKIPROWS = None

# Store observations in the spreadsheet in global variable
OBSERVATIONS = None

# List of entries in the observations, their label, and unit
OBS_ITEMS = dict()
OBS_ITEMS[ANG_RES]        = ['Angular resolution',       'arcsec']
OBS_ITEMS[BAND]           = ['Band',                     None]
OBS_ITEMS[DEC_DEG]        = ['Declination',              'dms']
OBS_ITEMS[FWHM_PB]        = ['12m primary beam size',    'arcsec']  # This is changed to 7m when printing for ACA Standalone SGs
OBS_ITEMS[IS_MOSAIC]      = ['Is mosaic?',               None]
OBS_ITEMS[IS_SPECTRAL]    = ['Is spectral scan?',           None]
OBS_ITEMS[LAS]            = ['Largest angular scale',    'arcsec']
OBS_ITEMS[LAT_OFFSET]     = ['Latitude offset',          'arcsec']
OBS_ITEMS[LON_OFFSET]     = ['Latitude offset',          'arcsec']
OBS_ITEMS[MAX_SIZE]       = ['Maximum field of view',    'arcsec']
OBS_ITEMS[MOS_AREA]       = ['Mosaic area',              'sq. arcmin']
OBS_ITEMS[MOS_COORD]      = ['Coordinate system',        None]
OBS_ITEMS[MOS_LENGTH]     = ['Length',                   'arcsec']
OBS_ITEMS[MOS_PA]         = ['Position angle',           'deg']
OBS_ITEMS[MOS_SPACING]    = ['Pointing spacings',        'arcsec']
OBS_ITEMS[MOS_WIDTH]      = ['Width',                    'arcsec']
OBS_ITEMS[POLARIZATION]   = ['Polarization',             None]
OBS_ITEMS[PROJECT]        = ['Project code',             None]
OBS_ITEMS[RA_DEG]         = ['Right ascension',          'hms']
OBS_ITEMS[REF_FREQ]       = ['Reference frequency',      'GHz']
OBS_ITEMS[REF_BW]         = ['Reference bandwidth',      'MHz']
OBS_ITEMS[REQ_SENS]       = ['Reference sensitivity',    'mJy']
OBS_ITEMS[SPS_START]      = ['Spectral scan start',      'GHz']
OBS_ITEMS[SPS_END]        = ['Spectral scan end',        'GHz']
OBS_ITEMS[SPS_BW]         = ['Spectral scan bandwidth',  'MHz']
OBS_ITEMS[SPS_SPW_RES]    = ['Spectral scan resolution', 'MHz']
OBS_ITEMS[TARGET]         = ['Target name',              None]
OBS_ITEMS[VELOCITY]       = ['Center velocity',          None]
OBS_ITEMS[VELOCITY_UNIT]  = ['Unit for velocity',        None]
OBS_ITEMS[VELOCITY_REF]   = ['Velocity frame',           None]
OBS_ITEMS[VELOCITY_DOP]   = ['Velocity convention',      None]
OBS_ITEMS[USE_7M]         = ['Use 7m array',             None]
OBS_ITEMS[USE_TPA]        = ['Use Total Power Array?',   None]

# Keywords for dictionary in readObservations only
DATA     = 'data'

# Keywords for dictionary in getSourceCoordinates()
RA       = 'ra'
DEC      = 'dec'
ORIGINAL = 'original'
PLOTSIZE = 'plotsize'
ISCOORD  = 'iscoord'
NOCOORD  = 'nocoord'

# Miscellaneous keywords
COORDS                = 'c'
MOSAIC_TYPE_RECTANGLE = 'Rectangle'
MOSAIC_TYPE_CUSTOM    = 'Custom'
MOSAIC_TYPES          = [MOSAIC_TYPE_RECTANGLE, MOSAIC_TYPE_CUSTOM]

# Miscellaneous parameters
EPSILON = 1e-4

# Set band frequencies
# http://www.almaobservatory.org/en/about-alma/how-does-alma-work/technology/front-end
BAND_UNKNOWN = 'unknown'
BAND_FREQ   = dict()
BAND_FREQ['ALMA_RB_01'] = [31, 50]
BAND_FREQ['ALMA_RB_02'] = [67, 90]
BAND_FREQ['ALMA_RB_03'] = [84, 116]
BAND_FREQ['ALMA_RB_04'] = [125, 163]
BAND_FREQ['ALMA_RB_05'] = [158, 211]
BAND_FREQ['ALMA_RB_06'] = [211, 275]
BAND_FREQ['ALMA_RB_07'] = [275, 373]
BAND_FREQ['ALMA_RB_08'] = [385, 500]
BAND_FREQ['ALMA_RB_09'] = [602, 720]
BAND_FREQ['ALMA_RB_10'] = [787, 950]
BAND_FREQ[BAND_UNKNOWN] = [0, 0]


def getSkipRows(filename):
   """ Rows to skip in LIST_OF_OBSERVATIONS file
       since they contain header information.
   """

   if filename.find(LIST_OF_OBSERVATIONS) == 0:
     skiprows = list(range(38))
     if len(skiprows) == 0:
        skiprows.append(1)
     else:
        skiprows.append(max(skiprows)+2)
   elif filename.find('archive') == 0:
     skiprows = []
   elif filename.find('proposals_cycle7') == 0:
     skiprows = []
   elif filename.find('proposals_cycle6') == 0:
     skiprows = []
   elif filename.find('proposals_cycle5') == 0:
     skiprows = []
   else:
     raise Exception('Unknown spreadsheet: %s' % filename)

   return skiprows


def getUsableBandwidth(bw, indx=None, throwException=True):
   """ 
       Returns the usable bandwidth in MHz.

       Inputs:
          bw  : nominal bandwidth in MHz.
          indx: Row number in the data structure. This is used for informational
                purposes only if there is an error in the input bw.
          throwException : If True, then throw an exception if bandwidth is invalid.
                           Otherwise, return None as the usable bandwidth.

       Output:
          usable bandwidth in MHz

       Notes:
          The usable bandwidths are given in Table 5.3 of the 
          Cycle 3 ALMA Technical Handbook.
   """
   msg = None

   if abs(bw - 62.5) < EPSILON or abs(bw - 58.6) < 0.1:
      use_bw = 58.6
   elif abs(bw - 125.) < EPSILON or abs(bw - 117.2) < 0.1:
      use_bw = 117.2
   elif abs(bw - 250.) < EPSILON or abs(bw - 234.4) < 0.1:
      use_bw = 234.4
   elif abs(bw - 500.) < EPSILON or abs(bw - 468.8) < 0.1:
      use_bw = 468.8
   elif abs(bw - 1000.) < EPSILON or abs(bw - 937.5) < 0.1:
      use_bw = 937.5
   elif abs(bw-1875) < EPSILON or abs(bw-2000) < EPSILON or (bw > 1875 and bw < 2000):
      use_bw = 1875.
   else:
      if throwException:
         msg = 'Do not recognize bandwidth value: %.1f' % bw
         if indx is not None:
            msg = 'Do not recognize bandwidth value (%.1f) on row %d' % (bw, getExcelFromIndex(indx))
      else:
         use_bw = None

   if msg is not None:
      raise Exception(msg)
   else:
      return use_bw


def checkEphemerisNames(data):
   """ 
       Check the names for sources that have a name "ephemeris". They
       should have been replaced with the actual source names from
       the proposals.
   """
#  print 'WARNING: Not checking ephemeris objects'
   return

   # Set codes/sources that are moving objects and names need to be modified.
   tuples = [ ]

   # Separate tuples
   found_tuples = [False] * len(tuples)
   codes = []
   sgnames = []
   for i in range(len(tuples)):
      codes.append(tuples[i][0])
      sgnames.append(tuples[i][1])
   codes   = np.array(codes)
   sgnames = np.array(sgnames)

   # Loop over rows in data structure
   for indx in data.index:
      # Get information
      code = data[PROJECT][indx]
      name = str(data[TARGET][indx])  # str is needed since some source names are numeric

      # Loop over cases
      if name.find('Ephemeris') == 0 and \
         abs(data[RA_DEG][indx]) < EPSILON and \
         abs(data[DEC_DEG][indx]) < EPSILON:
         # Find entry
         j = np.where( (codes == code) & (sgnames == name) )[0]
         if len(j) == 0:
            raise Exception('Could not find project code %s and source %s: see row=%d' % (code, name, getExcelFromIndex(indx)))
         elif len(j) > 1:
            raise Exception('Found multiple entries for project code %s and source %s' % (code, name))

         # Modify name
         data.loc[indx, (TARGET)] = tuples[j][2]
         found_tuples[j] = True

   # All rows should have been identified
   if found_tuples.count(False) > 0:
      print('Did not find the following ephemeris sources:')
      for j, found in enumerate(found_tuples):
         if not found:
            print('%s %s' % (tuples[j][0], tuples[j][1]))
      raise Exception('Not all ephemeris sources were identified.')


def getIndexFromExcel(excel, check=True):
   """ 
       Return the index number of the data structure from the excel row number.

       If check = True, the resultant values is checked for errors.
   """
   # Set index to account for header row and the index=1 in excel
   indx = excel - 2

   # Account for skipped rows. Assumes all skipped rows are before data
   if SKIPROWS is not None:
      indx -= len(SKIPROWS)

   # Check for errors
   lindex = makeList(indx)
   for l in lindex:
      if check and OBSERVATIONS is not None and \
         (l < 0 or l > OBSERVATIONS[DATA].index.max()):
         minRow = getExcelFromIndex(0, check=False)
         maxRow = getExcelFromIndex(OBSERVATIONS[DATA].index.max(), check=False)
         raise Exception('Excel row (%d) is out of range. Allowed range is between %d and %d' % \
            (excel, minRow, maxRow))

   return indx


def getExcelFromIndex(indx, check=True):
   """ 
       Return the row number in the excel spreadsheet from the index number
       in the data structure.

       If check = True, the resultant valueis checked for errors.
   """
   # Set index to account for header row and the index=1 in excel
   excel = indx + 2

   # Account for skipped rows. Assumes all skipped rows are before data
   if SKIPROWS is not None:
      excel += len(SKIPROWS)

   # Check for errors
   if check and OBSERVATIONS is not None: 
      minIndex = 0
      maxIndex = OBSERVATIONS[DATA].index.max()
      minExcel = getExcelFromIndex(minIndex, check=False)
      maxExcel = getExcelFromIndex(maxIndex, check=False)
      if excel < minExcel or excel > maxExcel:
         raise Exception('Index (%d) is out of range. Allowed range is between %d and %d' % \
            (indx, minIndex, maxIndex))

   return excel


def getBandNumber(band):
   """ 
       Returns band number from the band string in the spreadsheet.
       This assumes the band format is in ALMA_RB_NN.

       If there is an error, then 0 is return.
   """

   try :
      bn = int(band.split('_')[-1])
   except:
      bn = 0

   return bn


def makeList(sources, parse=True, delimiter=','):
    """ Convert the variable "sources" into a list.

        Input : sources   - a variable or list
                parse     - if True, parse elements with delimiter separator
                delimiter - the delimiter when parsing strings
        Output: Return value is a list.
                If input is a single value, then output is [sources]
                If input is a list, the result is copied

        Examples:
             (1) l = makeList('3c273')
             (2) l = makeList('3c273, 3c279')
             (3) l = makeList(['3c273','3c279'])
    """

    # Return if sources is empty
    if sources is None: return None

    # Convert to arrays
    l = list()
    t = [type(sources)]
    if str in t or str in t:
        l = [sources]
    else:
        try:
           i = len(sources)
           l = np.array(sources)
        except:
           l = np.array([sources])
    if len(l) == 0: return None

    # Parse string
    if parse:
        t = list()
        for z in l: 
             if str in [type(z)]: 
                for zz in z.split(delimiter):
                   t.append(re.sub(' +',' ', zz.strip()))
             else:
                 t.append(z)
        l = np.array(t[:])
    return l


def convertHmsString(value, ndec=0, showSeconds=True, delimiter=':'):
    """ 
        Converts floating point to HH:MM:[SS.S]

        Inputs 
           value       : floating point number
           ndec        : number of decimal points to print seconds
           showSeconds : If True, show seconds, otherwise just
                         use HH:MM. Default:True
           delimiter   : the characters to be used between the numbers.

        Output 
           a string of format hh:mm:[ss.s]

        Examples:
           convertHmsString(15.0)
           convertHmsString(15.0, ndec=2)
           convertHmsString(15.0, ndec=2, delimiter='hms')
           convertHmsString(15.0, ndec=2, delimiter='dms')
    """

    # Set delimiter
    spacing = delimiter
    if len(spacing) == 1: spacing = [spacing] * 3

    # Construct string
    t = value
    st = str(type(value))
    if st.find('int') >= 0 or st.find('float') >= 0:
        x = abs(value)
        h = int(x)
        m = int(60 * (x - h))
        sec = 3600.0 * (x - h - m/60.0)

        t = str("%.2d" % h) + spacing[0] + str('%.2d' % m) 

        if showSeconds  :
            # Add seconds
            t += spacing[1]
            format = '%0' + str(ndec+3) + '.' + str(ndec) + 'f'
            if ndec <= 0: format = '%.2d'
            t += str(format % sec)
            if spacing[2] not in [' ', ':']: t += spacing[2]

        if value < 0.0: t = '-' + t
    return t


def computeZ(data, indx):
   """ 
       Computes redshift based on velocity for row indx in data.
   """

   # Get velocity
   velocity      = data[VELOCITY][indx]
   velocity_unit = data[VELOCITY_UNIT][indx]
   doppler       = data[VELOCITY_DOP][indx]

   # Convert velocity to km/s
   if velocity_unit == 'km/s':
      v_kms = velocity
   elif velocity_unit == 'm/s':
      v_kms = velocity / 1e3
   else:
      raise Exception('Velocity unit not recognized: %s' % velocity_unit)

   # Compute redshift
   c_kms = G.c.value / 1e3
   if doppler == 'OPTICAL':
      z = v_kms / c_kms
   elif doppler == 'RADIO':
      z = v_kms/c_kms / (1.0 - v_kms/c_kms)
   elif doppler == 'RELATIVISTIC':
      z = np.sqrt( (1. + v_kms/c_kms) / (1.0 - v_kms/c_kms) ) - 1.0
   else:
      raise Exception('Doppler frame not recognized (%s) on row %d' % (doppler, getExcelFromIndex(indx)))

   # Done
   return z


def fwhmPB(freqGHz, diameter):
   """ 
       Returns primary FWHM in arcsec 

       freqGHz : frequency in GHz
       diameter: telescope diameter in meters

       See the knowledgebase article for more information:
         https://help.almascience.org/index.php?/Knowledgebase/Article/View/90/0/how-do-i-model-the-alma-primary-beam-and-how-can-i-use-that-model-to-obtain-the-sensitivity-profile-for-an-image-mosaic
   """
   return 1.13 * G.c.value / (freqGHz*1e9) / diameter / np.pi * 180.0 * 3600.0


def plotMosaic(ax, corners, fc='black', ec='None', 
               linewidth=2, alpha=0.5, hatch=None):
   """
        Plot a rectangular region for a mosaic.

        alpha     : transperancy of the plot rectangle
        ax        : pylab plot handle
        corners   : dictionary containing the corners of the mosaic, specific 
                    in RA,Dec offsets in arcseconds relative to the (0,0). It 
                    is best to use getMosaicCorners() to compute corner 
                    positions.
        ec        : edge color of the plot rectangle
        fc        : solid face color of the rectangle
        hatch     : hatch pattern for the rectangle
        linewidth : linewidth of the plot edge
   """
   UL = corners['UL']
   UR = corners['UR']
   BL = corners['BL']
   BR = corners['BR']

   xypolygon = np.zeros( (4, 2) )
   xypolygon[:,0] = np.array([UL[0], UR[0], BR[0], BL[0]])
   xypolygon[:,1] = np.array([UL[1], UR[1], BR[1], BL[1]])
   p = Polygon(xypolygon, closed=True, fc=fc, ec=ec, alpha=alpha, linewidth=linewidth, hatch=hatch)
   result = ax.add_artist(p)

#  ax.plot([UL[0], UR[0], BR[0], BL[0], UL[0]], 
#          [UL[1], UR[1], BR[1], BL[1], UL[1]], 
#          color=ec, linewidth=linewidth)

   return result


def plotPrimaryBeam(ax, xy, freq, diameter, fc='black', ec='black', 
                    alpha=1, linewidth=2):
   """
        Plot the primary beam FWHM for an observation at frequency freq in GHz
        for a telescope diameter in meters.

        ax        : pylab plot handle
        xy        : center of the primary beam
        fc        : face color of the plot circle
        ec        : edge color of the plot circle
        alpha     : transperancy of the plot circle
        linewidth : linewidth of the plot edge
   """
   radius = 0.5 * fwhmPB(freq, diameter)
   c = Circle( xy, radius=radius, fc=fc, ec=ec, alpha=alpha, linewidth=linewidth)
   result = ax.add_artist(c)
#  color = 'yellow'
#  e = Ellipse( (0, 0), width=2*radius, height=2*radius, angle=0,
#               color=color, fc=color, ec='black')
#  result = ax.add_artist(e)

   return result


def getBandColor(freq):
   """
       Set a plot color based on the input frequency (GHz),
       which a different color is chosen per ALMA band.
   """
   # Set band colors for plotting
   band_colors = dict()
   band_colors['ALMA_RB_01'] = 'gray'
   band_colors['ALMA_RB_02'] = 'silver'
   band_colors['ALMA_RB_03'] = 'blue'
   band_colors['ALMA_RB_04'] = 'peru'
   band_colors['ALMA_RB_05'] = 'yellow'
   band_colors['ALMA_RB_06'] = 'black'
   band_colors['ALMA_RB_07'] = 'green'
   band_colors['ALMA_RB_08'] = 'orange'
   band_colors['ALMA_RB_09'] = 'magenta'
   band_colors['ALMA_RB_10'] = 'red'
   band_colors[BAND_UNKNOWN] = 'tan'

   # Find color
   band_input = None
   for key, freq_range in BAND_FREQ.items():
      if freq >= freq_range[0] and freq <= freq_range[1]:
         band_input = key
         break
   if band_input is None:
      print('Warning: Frequency outside of ALMA bands.')
      band_input = BAND_UNKNOWN

   # Done
   return band_colors[band_input]


def checkObservations(obsdata, isdata=False):
   """ 
       Check that the user-supplied observation seems reasonable. 
       Only a light check is done.

       obsdata is either "observations" from readObservations (isdata=False) 
       or observations[DATA] (if isdata=True)
   """
   # Checks if isdata=False
   if isdata:
      # Must be panda structure
      if pd.core.frame.DataFrame not in [type(obsdata)]:
         raise Exception('observations must be a dictionary')

      # Get keys
      keys = obsdata.keys().tolist()
      msg = 'data'
   else:
      # Must have two keywords, if observations
      if dict not in [type(obsdata)]:
         raise Exception('observations must be a dictionary')

      # Check keys
      for key in [DATA, COORDS]:
         if key not in obsdata.has_key(key):
            raise Exception('%s keyword not found in observations' % key)

      # Get keys
      keys = obsdata[DATA].keys().tolist()
      msg = 'observations[%s]' % DATA

   # Check entries in OBS_ITEMS
   for key in OBS_ITEMS.keys():
      if keys.count(key) != 1:
         raise Exception('Keyword "%s" not found in %s' % (key, msg))

   # Check spectral keywords
   for w in SPW_FREQ.keys():
      if SPW_FREQ[w] not in keys:
         raise Exception('Keyword "%s" not found in %s' % (SPW_FREQ[w], msg))
      if SPW_BW[w] not in keys:
         raise Exception('Keyword "%s" not found in %s' % (SPW_BW[w], msg))
      if SPW_RES[w] not in keys:
         raise Exception('Keyword "%s" not found in %s' % (SPW_RES[w], msg))

   # Done
   return True


def checkData(data, verbose=True, spreadsheet=None):
   """ 
       Run checks on the data read from the Excel spreadsheet to
       catch any obvious anomalies.
   """
   # Message
   if verbose:
      print('Running checks on data')

   # Initialize
   error = False

   # Check keywords
   if verbose:
      print('   ... checking keywords in data structure')
   checkObservations(data, isdata=True)

   # Check RA
   if verbose:
      print('   ... checking right ascensions')
   j = np.where( (data[RA_DEG] < 0) | (data[RA_DEG] > 360.0) )[0]
   if len(j) > 0:
      error = True
      if verbose: print('        --- invalid right ascension in %d rows' % len(j))

   # Check Declination
   if verbose:
      if verbose: print('   ... checking declinations')
   j = np.where( (data[DEC_DEG] < -90.) | (data[DEC_DEG] > 90.0) )[0]
   if len(j) > 0:
      error = True
      if verbose:
         print('        --- invalid declination in %d rows' % len(j))

   # Check mosaic offsets
   if verbose:
      print('   ... checking mosaic parameters are present for rectangular mosaics')
   # Check mosaic system
   j = (data[MOS_COORD].isnull() == True) & (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE)
   if np.sum(j) > 0:
      error = True
      if verbose:
         print('        --- found %d mosaics where system was not set' % len(k))

   # Check PA
   j = (data[MOS_PA].isnull() == True) & (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE)
   if np.sum(j) > 0:
      error = True
      if verbose:
         print('        --- found %d mosaics where PA was not set' % np.sum(j))

   # Check length
   j = (data[MOS_LENGTH].isnull() == True) & (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE)
   if np.sum(j) > 0:
      error = True
      if verbose:
         print('        --- found %d mosaics where length was not set' % np.sum(j))
   j = (data[MOS_LENGTH] <= 0) & (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE)
   if np.sum(j) > 0:
      error = True
      if verbose:
         print('        --- found %d mosaics where length was zero' % np.sum(j))

   # Check width
   j = (data[MOS_WIDTH].isnull() == True) & (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE)
   if np.sum(j) > 0:
      error = True
      if verbose:
         print('        --- found %d mosaics where width was not set' % np.sum(j))
   j = (data[MOS_WIDTH] <= 0) & (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE)
   if np.sum(j) > 0:
      error = True
      if verbose:
         print('        --- found %d mosaics where width was zero' % np.sum(j))

   # Check if reference sensitivity is valid
   if verbose:
      print('   ... checking values of requested sensitivity are non-zero')
   mask = (data[REQ_SENS] <= 0.) | (np.isnan(data[REQ_SENS]) == True)
   if mask.sum() > 0:
      error = True
      if verbose: 
         print('        --- invalid sensitivity values in %d rows' % len(j))
         data.loc[mask, (REQ_SENS)] = 1.0

   # Check if reference bandwidth is valid
   if verbose:
      print('   ... checking values of reference bandwidth are non-zero')
   j = (data[REF_BW] <= 0.) | (data[REF_BW].isnull() == True)
   if np.sum(j) > 0:
      error = True
      print(data[PROJECT][j])
      if verbose: 
         print('        --- invalid reference bandwidth values in %d rows' % np.sum(j))

   # Check spectral scans
   if verbose:
      print('   ... checking spectral scan frequencies and bandwidths')
   for keys in [ [SPS_START, 'starting frequency'],
                 [SPS_END,   'ending frequency'],
                 [SPS_BW,   'bandwidth'],
                 [SPS_SPW_RES,   'spectral resolution'],
               ]:
      j = ((data[keys[0]].isnull() == True) | (data[keys[0]] <= 0)) & (data[IS_SPECTRAL] == True)
      if np.sum(j) > 0:
         error = True
         if verbose: 
            print('        --- spectral scan %s is not set in %d rows' % (keys[1], np.sum(j)))

   # Check that the frequencies match the bands
   if verbose:
      print('   ... checking values of frequencies and bandwidths')
   for indx in data.index:
      # Get range of frequencies defined this ALMA band
      freq_range = BAND_FREQ[data[BAND][indx]]

      # Check reference frequency
      if data[REF_FREQ][indx] < freq_range[0] or \
         data[REF_FREQ][indx] > freq_range[1]:
         error = True
         if verbose:
            print('        --- reference frequency out of range in row %d' % getExcelFromIndex(indx, check=False))

      # Check spectral scan frequencies
      if data[IS_SPECTRAL][indx]:
         # Starting frequency
         if data[SPS_START][indx] < freq_range[0] or \
            data[SPS_START][indx] > freq_range[1]:
            error = True
            if verbose:
               print('        --- spectral scan starting frequency out of range in row %d' % getExcelFromIndex(indx, check=False))

         # Ending frequency
         if data[SPS_END][indx] < freq_range[0] or \
            data[SPS_END][indx] > freq_range[1]:
            error = True
            if verbose:
               print('        --- spectral scan ending   frequency out of range in row %d' % getExcelFromIndex(indx))
      else:
         # Check each window
         for w in list(SPW_FREQ.keys()):
            # Check frequencies
            nu = data[SPW_FREQ[w]][indx]
            if np.isfinite(nu) and (nu < freq_range[0] or nu > freq_range[1]):
               error = True
               if verbose:
                  print('        --- spectral window frequency out of range in window %d on row %d' % \
                        (w, getExcelFromIndex(indx, check=False)))


            # Both frequencies and bandwidths need to be set
            bw = data[SPW_BW[w]][indx]
            if (np.isfinite(nu) and np.isnan(bw)) or \
               (np.isfinite(bw) and np.isnan(nu)):
               error = True
               if verbose:
                  print('        --- frequencies and bandwidths are not both set in window %d on row %d' % \
                        (w, getExcelFromIndex(indx, check=False)))

            # Check usable bandwidth
            if np.isfinite(nu) and getUsableBandwidth(bw, indx=indx) is None:
               error = True
               if verbose:
                  print('        --- bandwidth is not recognized in window %d on row %d' % \
                        (w, getExcelFromIndex(indx, check=False)))

   # Exit if there were any errors
   if verbose:
      print('   ... done')
   if error:
      if spreadsheet is None:
         msg = 'The spreadsheet contains unexpected errors.'
      else:
         msg = 'The spreadsheet %s contains unexpected errors.' % spreadsheet
      if not verbose:
         msg += '\nTry with verbose=True to see further information'
      raise Exception(msg)

   # Done
   return


def setSensitivitymJy(data, indx):
   """ Compute sensitivity in mJy """

   # Initialize
   unit = data[REQ_SENS_UNIT][indx]
   value = None

   # Loop over possible units
   if unit == 'mJy':
      value = data[REQ_SENS][indx]
   elif unit == 'Jy':
      value = data[REQ_SENS][indx] * 1e3
   elif unit in ['mK', 'K']:
      # Set scale factor to convert to mK
      scale = None
      if unit == 'mK':
         scale = 1.
      elif unit == 'K':
         scale = 1e3
      else:
         raise Exception('Un-recognized unit (%s) on row %d' % (unit, getExcelFromIndex(i)))

      # Get values needed to convert sensitivity to mJy
      tb_mK = data[REQ_SENS][indx] * scale
      angres = data[ANG_RES][indx]
      freq = data[REF_FREQ][indx]

      # Check values
      if tb_mK <= 0 or angres <= 0 or freq <= 0:
         print(indx)
         print(tb_mK, angres, freq)
         raise Exception('Error computing sensitivity on row %s' % (getExcelFromIndex(indx)))

      # Compute wavelength
      value = tb_mK * mjypermk(freq, angres, freq)
   else:
      raise Exception('Un-recognized unit (%s) on row %d' % (unit, getExcelFromIndex(indx)))

   # Check
   if value is None or value <= 0 or np.isnan(value):
      raise Exception('Error setting sensitivity (%s) for row %d' % (str(value), getExcelFromIndex(indx)))

   # Done
   return value


def getFinestResolution(data, indx):
   """ Return finest resolution in the spectral setup in MHz """
   if data[IS_SPECTRAL][indx]:
      resolution = data[SPS_SPW_RES][indx]
   else:
      # Get frequencies of each window
      resolution = None
      for w in list(SPW_FREQ.keys()):
         # Is this a valid window?
         if np.isnan(data[SPW_FREQ[w]][indx]): 
            continue

         # Set resolution
         res = data[SPW_RES[w]][indx]

         # Get minimum
         if resolution is None or res < resolution:
            resolution = res

   # Check
   if resolution is None or resolution <= 0 or np.isnan(resolution):
      raise Exception('Error setting resolution for row %d' % getExcelFromIndex(indx))

   # Done
   return resolution


def getLargestBandwidth(data, indx):
   """ Return largest bandwidth in the spectral setup in MHz """
   if data[IS_SPECTRAL][indx]:
      bandwidth = data[SPS_BW][indx]
   else:
      # Get frequencies of each window
      bandwidth = None
      for w in list(SPW_FREQ.keys()):
         # Is this a valid window?
         if np.isnan(data[SPW_FREQ[w]][indx]): 
            continue

         # Set resolution
         bw = data[SPW_BW[w]][indx]

         # Get minimum
         if bandwidth is None or bw > bandwidth:
            bandwidth = bw

   # Check
   if bandwidth is None or bandwidth <= 0 or np.isnan(bandwidth):
      raise Exception('Error setting largest bandwidth for row %d' % getExcelFromIndex(indx))

   # Done
   return bandwidth


def setReferenceBandwidth(data, indx):
   """ Set reference bandwidth in Hz """

   # Initialize
   measure = data[REF_BW_MEASURE][indx]
   value   = data[REF_BW][indx]

   # Loop over possible measures
   if measure == 'AggregateBandWidth':
      value = computeAggregateBandwidth(data, indx)
   elif measure == 'FinestResolution':
      value = getFinestResolution(data, indx)
   elif measure == 'LargestWindowBandWidth':
      value = getLargestBandwidth(data, indx)
   elif measure == 'RepresentativeWindowBandWidth':
      value = data[REP_WIN_BW][indx]
   elif measure == 'RepresentativeWindowResolution':
      value = data[REP_WIN_RES][indx]
   elif measure == 'User':
      pass
   else:
      raise Exception('Un-recognized bandwidth measure (%s) on row %d' % (measure, getExcelFromIndex(indx)))

   # Check
   if value is None or value <= 0 or np.isnan(value):
      raise Exception('Error setting reference bandwidth for row %d' % getExcelFromIndex(indx))

   # Done
   return value


def computeCoords(data):
   """ Computes astropy coordinates """
   coords = SkyCoord(data[RA_DEG].values*astropy.units.degree, 
                     data[DEC_DEG].values*astropy.units.degree, frame='icrs')

   return coords


def computeMeanFrequency(data, indx):
   """ 
       Computes mean frequency of a correlator setup. 
       In computing the mean frequency, the frequency is weighted by
       the bandwidth.
   """
   if data[IS_SPECTRAL][indx]:
      meanfreq = 0.5 * (data[SPS_START][indx] + data[SPS_END][indx])
   else:
      # Initialize
      sumf  = 0.
      sumbw = 0.

      # Loop over spectral windows
      for win, key in SPW_FREQ.items():
         # Skip if not a valid window
         if np.isnan(data[key][indx]):
            continue

         # Sum
         nu = data[key][indx]
         bw = data[SPW_BW[win]][indx]
         sumf  += nu*bw
         sumbw += bw

      # Compute mean frequency
      if sumbw == 0:
         raise Exception('Bug for indx=%d, %s, %s...' % \
                  (indx, data[PROJECT][indx], data[TARGET]))
      meanfreq = sumf / sumbw
   
   # Done
   return meanfreq


def readObservations(input=LIST_OF_OBSERVATIONS, verbose=True, 
                    removeSolar=True, removeVLBI=True, 
                    onlySolar=False, correctMosaicSpacing=True,
                    clearSolar=False):
   """ Wrapper to readObservation() """

   # Initialize data structure
   data = None

   # Loop over input files
   for fn in makeList(input):
      # Read
      tmp = readObservingFile(fn, verbose=verbose, removeSolar=removeSolar, removeVLBI=removeVLBI, onlySolar=onlySolar, correctMosaicSpacing=correctMosaicSpacing, clearSolar=clearSolar)

      # Concatenate files
      if data is None:
         data = tmp
      else: 
         data = data.append(tmp, ignore_index=True)

   # Compute RA/DEC
   if verbose:
      print('   ... converting RA/DEC to astropy sky coordinates')
   coords = computeCoords(data)

   # Done
   print('NOTE: This script only checks on-going observations.')
   print('      Completed observations are listed in the ALMA archive.')
   return {DATA: data, COORDS:coords}


def readObservingFile(input=LIST_OF_OBSERVATIONS, verbose=True, 
                      removeSolar=True, removeVLBI=True, onlySolar=False,
                      correctMosaicSpacing=False, clearSolar=False):
   """
       Read in observations using pandas.

       A global variable called SKIPROWS is set indicating which rows
       were skipped in reading the data file.

       Inputs:
          input : root name for the file containing the existing and 
                  scheduled observations. The .csv or .xlsx extension can
                  be omitted.
          correctMosaicSpacing : If True, corrects an error in Ignacio's 
                  spreadsheet where the mosaic spacing was too fine.
          clearSolar : If True, set coordinates of solar observations to (0,0) and set name to Sun.

       Output:
          A python dictionary with the following entries:
            DATA      : a panda data set containing the spreadsheet
            COORDS    : astropy sky coordinates for each source in DATA
   """
   # Read in observations. Input list may either be an excel spreadsheet or 
   # a csv file. Both are tried, with the .csv first and then the .xlsx.
   data = None
   input_without_extension = input.replace('.xlsx','').replace('.xls', '').replace('.csv', '')
   if verbose:
      print('Reading observations')
   for ext in ['csv', 'xls', 'xlsx']:
      # Set file name
      inputFile = '%s.%s' % (input_without_extension, ext)

      # Get number of rows to skip
      skiprows = getSkipRows(inputFile)

      # Try reading file
      if os.path.exists(inputFile):
         if ext == 'csv':
            data = pd.read_csv(inputFile, skiprows=skiprows, header=0, low_memory=False)
         elif ext in ['xls', 'xlsx']:
            data = pd.read_excel(inputFile, skiprows=skiprows, header=0)
         else:
            raise Exception('Do not recognize extension for file %s' % input)

      # Did we succeed?
      if data is not None: 
         if verbose:
            print('   ... read %d rows in %s' % (data.shape[0], inputFile))
            global SKIPROWS
            SKIPROWS = skiprows
         break

   # Check if data were read
   if data is None:
      raise Exception('Could not read data. Check that the input file %s exists with either a .csv for .xls extension.' % input)

   # Delete column containing RA/DEC in string form since RA/DEC may
   # be modified below.
   del data[RA_HMS]
   del data[DEC_DMS]

   # Copy RA/DEC to original values before apply mosaic
   data[RA_DEG_ORIG]  = data[RA_DEG]
   data[DEC_DEG_ORIG] = data[DEC_DEG]

   # Set variable indicating if observation is a spectral scan
   data[IS_SPECTRAL] = (data[SPS_START].notnull() == True)

   # Add flag if ACA standalone
   if not IS_ACA_STANDALONE in data.columns:
      print('   ... WARNING: assuming these are not ACA standalone observations')
      data[IS_ACA_STANDALONE] = np.array([False] * data.shape[0])
#  else:
#     data[USE_12M] = np.array([True] * data.shape[0])

   # Remove (or include) solar observations, if needed. If we are not removing 
   # them, then we need to check sensitivities and bandwidth are non-zero.
   if IS_SOLAR in data.columns:
      if removeSolar:
         # Find how many there are
         nrows     = data[data[IS_SOLAR] == True].shape[0]
         nprojects = data[PROJECT][data[IS_SOLAR] == True].unique().shape[0]
         print('   ... removing %d solar observations in %d projects' % (nrows, nprojects))
         data = data[data[IS_SOLAR] == False]
      else:
         # Reset if needed
         if clearSolar:
            # Message
            print('   ... resetting coordinates and names for solar observations')

            # Get solar observations
            mask = (data[IS_SOLAR])

            # Reset name
#           data.loc[mask, TARGET] = 'Sun'

            # Reset coordinates
            data.loc[mask, RA_DEG] = 0
            data.loc[mask, DEC_DEG] = 0
            data.loc[mask, RA_DEG_ORIG]  = 0
            data.loc[mask, DEC_DEG_ORIG] = 0

         # Check sensitivity
         mask = (data[IS_SOLAR] == True) & (data[REQ_SENS] == 0)
         data.loc[mask, REQ_SENS] = 0.1

         # Check bandwidth
         mask = (data[IS_SOLAR] == True) & (data[REF_BW]==0)
         data.loc[mask, REF_BW] = 7500.0

         # Keep only solar if needed
         if onlySolar:
            print('Only keeping solar observations')
            data = data[data[IS_SOLAR] == True]

   # Remove VLBI observations, if needed. If we are not removing them, 
   # then we need to check sensitivities and bandwidth are non-zero.
   if IS_VLBI in data.columns:
      if removeSolar:
         # Find how many there are
         nrows     = data[data[IS_VLBI] == True].shape[0]
         nprojects = data[PROJECT][data[IS_VLBI] == True].unique().shape[0]
         print('   ... removing %d VLBI observations in %d projects' % (nrows, nprojects))
         data = data[data[IS_VLBI] == False]
      else:
         # Check sensitivity
         j = (data[IS_VLBI] == True) & (data[REQ_SENS]==0)
         data.loc[j, REQ_SENS] = 0.1

         # Check bandwidth
         j = (data[IS_VLBI] == True) & (data[REF_BW]==0)
         data.loc[j, REF_BW] = 7500.0

         # Check angular resolution
         j = (data[IS_VLBI] == True) & (data[ANG_RES]==0)
         data.loc[j, ANG_RES] = 0.00001


   # Convert sensitivity to mJy
   if REQ_SENS_UNIT in data.columns:
      # Initialize
      sens_mjy = np.zeros(data.shape[0])

      # Recompute sensitivity
      for i, indx in enumerate(data.index):
         sens_mjy[i] = setSensitivitymJy(data, indx)

      # Save
      data.loc[:, (REQ_SENS)] = sens_mjy

      # Deletet unit column
      del data[REQ_SENS_UNIT]

   # Convert reference frequency width to MHz
   if REF_BW_UNIT in data.columns:
      # Only support GHz
      nrows = data[data[REF_BW_UNIT] != 'GHz'].shape[0]
      if nrows > 0:
         raise Exception('%d values for reference frequency bandwidth are not GHz' % nrows)

      # Convert to MHz
      data[REF_BW] *= 1000

      # Delete unit
      data[REF_BW_UNIT]

   # Set reference bandwidth
   if REF_BW_MEASURE in data.columns:
      # Initialize
      ref_bw = np.array(data[REF_BW].values, dtype=float)

      # Set reference bandwidth
      for i, indx in enumerate(data.index):
#        if ref_bw[i] <= 0:
         ref_bw[i] = setReferenceBandwidth(data, indx)

      # Save
      data.loc[:, (REF_BW)] = ref_bw

      # Deletet measure column
      del data[REF_BW_MEASURE]

   # Set science goal, if needed
   if SG_NAME not in data.columns:
      data[SG_NAME] = np.array([''] * data.shape[0], dtype='str')
   if SG_FULLNAME not in data.columns:
      data[SG_FULLNAME] = np.array([''] * data.shape[0], dtype='str')

   # Set PI name to blank if column is not present
   if PINAME not in data.columns:
      data[PINAME] = np.array([''] * data.shape[0], dtype='str')

   # Set moving objects
   data[MOVING_OBJECT] = (data[RA_DEG_ORIG].abs() < EPSILON) & \
                         (data[DEC_DEG_ORIG].abs() < EPSILON)

   # Set large programs
   data[IS_LP] = data[PROJECT].str.endswith(".L")

   # Set excel line
   cnst = 2
   if skiprows is not None:
      cnst += len(skiprows)
   data[EXCEL_LINE] = data.index + cnst

   # Correct the RA/DEC for sources with an RA/DEC offset. Note the offsets
   # may be given in galactic coordinates for mosaics. 
   if verbose:
      print('   ... correcting centroid RA/DEC in mosaics for offsets')
   # Assume J2000 if no coordinate system provided
   j = (data[MOS_COORD].isnull() == True)
   data.loc[j, MOS_COORD] = 'J2000'
   # Get unique coordinate sysmtes
   coord_system = data[MOS_COORD][data[MOS_COORD].notnull() == True].unique()
   # Correction depends on the coordinate system
   for cs in coord_system:
      # Find observations with this system
      j = (data[MOS_COORD] == cs) & \
          (data[LAT_OFFSET].notnull() == True) & \
          (data[LON_OFFSET].notnull() == True) & \
          ( (data[LON_OFFSET].abs() > EPSILON) |
            (data[LAT_OFFSET].abs() > EPSILON))

      # Get data
      x  = data.loc[j, RA_DEG].values
      y  = data.loc[j, DEC_DEG].values
      dx = data.loc[j, LON_OFFSET].values
      dy = data.loc[j, LAT_OFFSET].values

      # Correct coordinates
      if cs in ['ICRS', 'icrs', 'J2000']:
         # Message
#        if verbose:
#           print '       --- correcting %4d positions for equatorial offsets' % len(j)

         # Correct RA/Dec
         new_ra  = x + dx / 3600.0 / np.cos(y / 180.0 * np.pi)
         new_dec = y + dy / 3600.0
         pass
      elif cs == 'galactic':
         # Message
#        if verbose:
#           print '       --- correcting %4d mosaic positions for galactic offsets' % np.sum(j)

         # Convert mosaic center to galactic coordinates
         c = SkyCoord(ra=x, dec=y, frame='icrs', unit='degree')
         glon = c.galactic.l.deg
         glat = c.galactic.b.deg

         # Add offset
         glon += dx / 3600.0 / np.cos(glat / 180.0 * np.pi)
         glat += dy / 3600.0

         # Convert back to RA/DEC
         c = SkyCoord(glon, glat, unit='degree', frame='galactic')
         new_ra  = c.icrs.ra.deg
         new_dec = c.icrs.dec.deg
      else:
         raise Exception('Mosaic coordinate system not implemented: %s' % cs)

      # Check right ascension
      l = np.where(new_ra < 0)[0]
      if len(l) > 0:
         new_ra[l] += 360.0
      l = np.where(new_ra > 360)[0]
      if len(l) > 0:
         new_ra[l] -= 360.0

      # Check declination
      if np.any(np.abs(new_dec) > 90.0):
         raise Exception('Error setting mosaic declination')

      # Save
      data.loc[j, RA_DEG]  = new_ra
      data.loc[j, DEC_DEG] = new_dec

   # Convert RA/DEC to degree in sky coordinates
#  if verbose:
#     print '   ... converting RA/DEC to astropy sky coordinates'
#  coords = SkyCoord(data[RA_DEG]*astropy.units.degree, 
#                    data[DEC_DEG]*astropy.units.degree, frame='icrs')

   # Set the velocity unit.
   # Until now, all SG have used km/s as the velocity unit. but I want to keep 
   # the flexibility in case m/s is used in the future.
   if VELOCITY_UNIT in data.columns:
      nrows = data[data[VELOCITY_UNIT] != 'km/s'].shape[0]
      if nrows > 0:
         raise Exception('%d values for velocity unit are not km/s' % nrows)
   else:
      data[VELOCITY_UNIT] = np.array(['km/s'] * data.shape[0])

   # Compute redshift
   if verbose:
      print('   ... computing redshifts')
   redshifts = np.zeros(data.shape[0])
   for i, indx in enumerate(data.index):
      redshifts[i] = computeZ(data, indx)

   # Correct frequencies for spectral windows that have rest frequencies
   if verbose:
      print('   ... correcting rest frequencies to sky frequencies')
   jndx = dict()
   znu  = dict()
   for w in list(SPW_FREQ.keys()):
      jndx[w] = []
      znu[w]  = []
   for i, indx in enumerate(data.index):
      # Determine if windows are doppler corrected or not
      if data[IS_SPECTRAL][indx]:
         if data[IS_SPW_SKY_FREQ][indx] == True:
            pass
         else:
            raise Exception('Not expecting spectral scan to be rest frequencies: excel line = %d' % getExcelFromIndex(indx, check=False))
      else:
         # Determine if windows are sky or rest frequencies
         if data[IS_SPW_SKY_FREQ][indx] not in [0, 1, False, True]:
            raise Exception('Error reading IS_SPW_SKY_FREQ on row %d' % \
               getExcelFromIndex(indx))
         elif not data[IS_SPW_SKY_FREQ][indx]:
            # Loop over windows and correct frequencies
            for w in list(SPW_FREQ.keys()):
               nu = data[SPW_FREQ[w]][indx]
               if np.isfinite(nu):
                  jndx[w].append(indx)
                  znu[w].append(nu / (1.0 + redshifts[i]))
   for w in list(SPW_FREQ.keys()):
      if len(jndx[w]) > 0:
         data.loc[jndx[w],(SPW_FREQ[w])] = znu[w]

   # Since all windows are now doppler corrected, delete IS_SKY_FREQ columns
   del data[IS_SPW_SKY_FREQ]

   # If reference frequency is zero, then set it to the mean frequency
   # This has to be done AFTER correcting the frequencies for the doppler shift
   j = (data[REF_FREQ] == 0)
   n = np.sum(j)
   if n > 0:
      # Print message
      print('   ... correcting reference frequency on %d rows' % n)
 
      # Make sure required variables are present
      meanfreq = np.zeros(n)
      for i, indx in enumerate(j[j==True].index):
         meanfreq[i] = computeMeanFrequency(data, indx)
      data.loc[j, REF_FREQ] = meanfreq

   # Get FWHM primary beam size
   mask = (data[REF_FREQ] < 0) | (data[REF_FREQ].isnull())
   if mask.sum() > 0:
      raise Exception('Invalid frequencies in spreadsheet')
   diameter = np.array([12.] * data.shape[0])
   j = np.where( (data[IS_ACA_STANDALONE] == True) )
   diameter[j] = 7.0
   data[FWHM_PB] = fwhmPB(data[REF_FREQ], diameter)

   # Set largest size in arcseconds, including mosaics and FWHM of primary beam
   data[MAX_SIZE] = data[ [MOS_LENGTH, MOS_WIDTH, FWHM_PB] ].max(axis=1)

   # If IS_MOSAIC is all nan, then convert it to string
   n = data[IS_MOSAIC][data[IS_MOSAIC].notnull()].count()
   if n == 0:
     data[IS_MOSAIC] = np.array(['N/A'] * data.shape[0])

   # Compute mosaic area
   if verbose:
      print('   ... computing area of rectangular mosaics')
   data[MOS_AREA] = np.zeros_like(data[MOS_LENGTH])
   mask = (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE)
   if mask.sum() > 0:
      print('   ... computing mosaic area of %d mosaics' % mask.sum())
      data.loc[mask, MOS_AREA] = data[MOS_LENGTH][mask] * data[MOS_WIDTH][mask] / 3600.0

   # Get source name with mosaic extension for custom mosaics
   names = [''] * data.shape[0]
   for j, indx in enumerate(data.index):
      # Get target name in spread sheet
      name = str(data[TARGET][indx])

      # Is this a custom mosaic?
      if isObsMosaic(data, indx, mtype=MOSAIC_TYPE_CUSTOM) and name.find('None') != 0:
         # Get name without the extension
         i = name.rfind('-')
         if i <= 0 or not name.endswith('(cm)'):
            raise Exception('Unknown extension for custom mosaic: %s on row %d' % (name, getExcelFromIndex(indx)))
         name = name[:i]

      # Save
      names[j] = name
   data[TARGET_STRIPPED] = names

   # Correct mosaic spacings.
   # The ACA mosaic spacings are incorrect in Ignacio's spreadsheet. I do not
   # the correct spacing, so this is potentially in error.
   if correctMosaicSpacing:
      # Message
      print('   ... correcting ACA mosaic spacings')

      # Select data
      mask = (data[IS_MOSAIC] == MOSAIC_TYPE_RECTANGLE) & (data[IS_ACA_STANDALONE])

      # Correct
      data.loc[mask, MOS_SPACING] = 0.51093 * data[FWHM_PB][mask]

   # Modify ephemeris names
   if verbose:
      print('   ... checking names for ephemeris sources')
   checkEphemerisNames(data)

   # Message
   if verbose:
      print('   ... done')

   # Run checks on the data structure
   checkData(data, verbose=verbose, spreadsheet=input)

   # Done
#  return {DATA: data, COORDS:coords}
   return data


def computeAggregateBandwidth(data, indx):
   """ Compute aggregate bandwidth in MHz for indx """
   if data[IS_SPECTRAL][indx]:
      sps_bw = getUsableBandwidth(data[SPS_BW][indx])
      nwin = 4
      aggregateBandwidth = nwin * sps_bw
   else:
      # Get frequencies of each window
      windows = list(SPW_FREQ.keys())
      nu1 = np.zeros(len(windows))
      nu2 = np.zeros(len(windows))
      use = np.zeros(len(windows), dtype=bool)
      for i, w in enumerate(windows):
         # Is this a valid window?
         if np.isnan(data[SPW_FREQ[w]][indx]): 
            use[i] = False
            continue
         freq = data[SPW_FREQ[w]][indx] * 1e3
         bw   = getUsableBandwidth(data[SPW_BW[w]][indx])
         use[i] = True
         f1     = freq - 0.5 * bw
         f2     = freq + 0.5 * bw
         nu1[i] = min([f1, f2])
         nu2[i] = max([f1, f2])

      # Sort by decreasing bandwidth
      j = np.where(nu1 > 0)
      nu1 = nu1[j]
      nu2 = nu2[j]
      bw = nu2 - nu1
      iarg = np.argsort(bw)[::-1]
      nu1 = nu1[iarg]
      nu2 = nu2[iarg]

      # Compute aggregate bandwidth
      j = np.where(nu1 > 0)
      for iw in range(nu1.size):
         # Skip if not using the window
         if not use[iw]: 
            continue

         # Check other windows
         for jw in range(iw+1, nu1.size):
            # Skip name windows
            if not use[jw]: 
               continue

            # Check frequency range
            if nu1[jw] >= nu1[iw] and nu2[jw] <= nu2[iw]:
               nu1[jw] = 0
               nu2[jw] = 0
               use[jw] = False
            elif nu1[jw] < nu1[iw] and nu2[jw] > nu2[iw]:
               raise Exception('Should not be here since I sorted by decreasing bandwidth. Indx=%d' % indx)
            elif nu1[jw] >= nu2[iw] or nu2[jw] <= nu1[iw]:
               pass
            elif nu1[jw] >= nu1[iw] and nu1[jw] <= nu2[iw] and nu2[jw] >= nu2[iw]:
               nu1[jw] = nu2[iw]
               if nu1[jw] == nu2[jw]:
                  use[jw] = False
               elif nu2[jw] < nu1[jw]:
                  raise Exception('Error setting bandwidth')
            elif nu1[jw] <= nu1[iw] and nu2[jw] >= nu1[iw] and nu2[jw] <= nu2[iw]:
               nu2[jw] = nu1[iw]
               if nu1[jw] == nu2[jw]:
                  use[jw] = False
               elif nu2[jw] < nu1[jw]:
                  raise Exception('Error setting bandwidth')
            else:
               raise Exception('Error computing aggregate bandwidth')

      # Compute aggregate bandwidth in MHz
      aggregateBandwidth = np.sum(nu2 - nu1)

   # Done
   return aggregateBandwidth


def computeContinuumSensitivity(data, indx):
   """ 
        Compute the continuum sensitivity for a single pointing 

        For spectral scans, the sensitivity is computed for one tuning,
        not the full spectral scan.

        Inputs:
           data : data structure from readObservations()
           indx : Index number in data

        Output:
           A dictionary with the following keys:
           'bw'    : aggregrate bandwidth in MHz
           'rmsmJy': continuum rms in mJy
           'rmsmK' : continuum rms in mK for the specified angular 
                     resolution and reference refrequencin data
   """

   # Gather sensitivity information
   req_sen  = data[REQ_SENS][indx]
   ref_freq = data[REF_FREQ][indx]
   ref_bw   = data[REF_BW][indx]
   angres   = data[ANG_RES][indx]

   # Compute bandwidth if single tuning or spectral scans
   aggregateBandwidth = computeAggregateBandwidth(data, indx)

   # Compute continuum sensitivity
   if ref_bw <= 0 or aggregateBandwidth <= 0:
      rmsContinuum = np.nan
      raise Exception('Error computing continuum sensitivity')
   else:
      rmsContinuum_mJy = req_sen * np.sqrt(ref_bw / aggregateBandwidth)
      rmsContinuum_mK  = rmsContinuum_mJy  / mjypermk(ref_freq, angres, ref_freq)

   # Done
   return {'bw': aggregateBandwidth, 'rmsmJy': rmsContinuum_mJy, 
           'rmsmK': rmsContinuum_mK}


def printRowProject(data, indx, spaces=''):
   """
       Print project information for entry indx in "data".
   """

   # Functions to print data
   def printEntries(obsdata, indx, items):
      for key in items:
         a = OBS_ITEMS[key]
         printEntry(obsdata, indx, key, a[0], a[1])

   def printEntry(obsdata, indx, key, label, unit):
      # Set unit
      sunit = unit
      if sunit is None: sunit = ''

      # Set value - there must be a better way!
      if key in [FWHM_PB, REF_FREQ, REF_BW]:
         svalue = '%.1f' % obsdata[key][indx]
         # If ACA standalone label for primary beam is changed from 12m to 7m
         if obsdata[IS_ACA_STANDALONE][indx]:
             label = label.replace('12m','7m')
      elif key in [MOS_LENGTH, MOS_WIDTH]:
         svalue = '%.1f' % obsdata[key][indx]
      elif key in [MOS_AREA]:
         svalue = '%.3f' % obsdata[key][indx]
      elif key in [REQ_SENS, ANG_RES]:
         svalue = '%.3f' % obsdata[key][indx]
      elif key in [RA_DEG]:
         svalue = convertHmsString(obsdata[key][indx] / 15.0, delimiter=':', ndec=2)
      elif key in [DEC_DEG]:
         svalue = convertHmsString(obsdata[key][indx], delimiter=':', ndec=2)
      elif key == BAND:
         svalue = '%d' % getBandNumber(obsdata[key][indx])
      elif key in [POLARIZATION]:
         svalue = obsdata[key][indx].lower()
      elif key in [MOS_SPACING]:
         svalue = '%.1f' % obsdata[key][indx]
      elif key == IS_MOSAIC:
         if isObsMosaic(obsdata, indx, mtype=MOSAIC_TYPE_RECTANGLE):
            svalue = 'Rectangular mosaic'
         elif isObsMosaic(obsdata, indx, mtype=MOSAIC_TYPE_CUSTOM):
            svalue = 'Custom mosaic'
         elif obsdata[IS_MOSAIC][indx] == 'N/A' or np.isnan(obsdata[IS_MOSAIC][indx]):
            svalue = 'False'
         else:
            raise Exception('Unknown type of mosaic: %s' % data[IS_MOSAIC][indx])
      else:
         svalue = '%s' % obsdata[key][indx]

      # Print item
      print('%s%-22s  %-13s  %s' % (spaces, label, svalue, sunit))

   # Add some space
   print('')
   print('')

   # Print header
   lineExcel = getExcelFromIndex(indx)
   print('%sSource information for spreadsheet line %d (project = %s)' % \
      (spaces, lineExcel, data[PROJECT][indx]))

   # Print source information
   items = [TARGET,
            RA_DEG,
            DEC_DEG,
          ]
   printEntries(data, indx, items)

   # Print observing parameters
   print('')
   print('%sObserving parameters' % spaces)
   items = [BAND,
            FWHM_PB,
            ANG_RES,
            LAS,
           ]
   printEntries(data, indx, items)

   # Print observing modes
   print('')
   print('%sObserving modes' % spaces)
   items = [POLARIZATION, USE_7M, USE_TPA, IS_SPECTRAL]
   printEntries(data, indx, items)

   # Print mosaic information
   print('')
   print('%sMosaic information' % spaces)
   items = [IS_MOSAIC]
   if isObsMosaic(data, indx, mtype=MOSAIC_TYPE_RECTANGLE):
      items.extend([MOS_AREA, MOS_LENGTH, MOS_WIDTH, MOS_PA, MOS_SPACING, MOS_COORD])
   printEntries(data, indx, items)

   # Print continuum sensitivity if not a spectral scan
   if not data[IS_SPECTRAL][indx]:
      print('')
      print('%sEstimated continuum sensitivity' % spaces)
      result = computeContinuumSensitivity(data, indx)
      print('%s%-22s  %-13.0f  %s' % \
         (spaces, 'Aggregate bandwidth', result['bw'], 'MHz'))
      if np.isnan(result['rmsmJy']):
         print('%s%-22s  %-13s    %s' % \
            (spaces, 'Continuum RMS', 'N/A', ''))
      else:
         print('%s%-22s  %-13.3f  %s' % \
         (spaces, 'Continuum RMS', result['rmsmJy'], 'mJy'))
         print('%s%-22s  %-13.1f  %s' % \
         (spaces, 'Continuum RMS', result['rmsmK'], 'mK'))


def mjypermk(freq_ghz, angres_ref, freq_ref_ghz):
   """ 
       Returns the conversion factor to convert mK to mJy.
   """

   omega_ref  = np.pi / 4.0 / np.log(2.0) * (angres_ref / 3600.0 / 180.0 * np.pi)**2
   omega      = omega_ref * (freq_ref_ghz / freq_ghz)**2
   wavelength = G.c.value / (freq_ghz * 1e9)
   constant   = 2.0 * G.k_B.value * 1e-3 / wavelength**2 * omega * 1e29

   return constant

def printRowCorrelator(data, indx, spaces=''):
   """
       Print correlator configuration for a row entry observations.
       The output will be different for spectral scans and single tunings.

       Inputs:
          data   : Contains data from readObservations() in the DATA key
          indx   : Entry number in observations
          spaces : Spaces to format output
   """

   if data[IS_SPECTRAL][indx]:
      printRowCorrelatorSpectralScan(data, indx, spaces=spaces)
   else:
      printRowCorrelatorSingle(data, indx, spaces=spaces)


def printRowCorrelatorSingle(data, indx, spaces=''):
   """
       Print project information for entry indx in "observations" if it
       is a single tuning.

       Inputs:
          data   : Contains data from readObservations() in the DATA key
          indx   : Entry number in observations
          spaces : Spaces to format output
   """

   # Add some space
   print('')

   # Print header
   lineExcel = getExcelFromIndex(indx)
   print('%sCorrelator setup' % spaces)

   # Print header
   print('%s%3s %8s  %17s  %19s  %17s  %17s' % \
      (spaces, 'Win', 'Sky Freq', 'Usable Bandwidth', 'Spectral resolution', 
       'RMS/bandwidth', 'RMS/resolution'))
   print('%s%3s %8s  %17s  %19s  %17s  %17s' % \
      (spaces, '---', '--------', '-----------------', '-------------------', 
       '----------------', '-----------------'))
   print('%s%3s %8s  %8s %8s  %9s %9s  %8s %8s  %8s %8s' % \
      (spaces, '', '(GHz)', '(MHz)', '(km/s)', '(MHz)', '(km/s)',
       'mJy', 'mK', 'mJy', 'K'
      ))

   # Gather sensitivity information
   req_sen  = data[REQ_SENS][indx]
   ref_freq = data[REF_FREQ][indx]
   ref_bw   = data[REF_BW][indx]

   # Get frequencies
   windows = list(SPW_FREQ.keys())
   winfreq = []
   winnumber = []
   for w in windows:
      if np.isfinite(data[SPW_FREQ[w]][indx]): 
         winnumber.append(w)
         winfreq.append(data[SPW_FREQ[w]][indx])

   # Sort by frequency
   iarg = np.argsort(winfreq)
   winfreq = np.array(winfreq)[iarg]
   winnumber = np.array(winnumber)[iarg]

   # Loop over windows
   for w in winnumber:
      # Is this a valid window?
      if np.isnan(data[SPW_FREQ[w]][indx]): 
         continue

      # Gather spectral window information
      freq  = data[SPW_FREQ[w]][indx]
      bw    = getUsableBandwidth(data[SPW_BW[w]][indx])
      res   = data[SPW_RES[w]][indx]

      # Print window
      print('%s%3d' % (spaces, w), end=' ')

      # Frequency
      print('%8.3f' % freq, end=' ')

      # Bandwidth in MHz
      print(' %8.1f' % bw, end=' ')

      # Bandwidth in km/s
      bw_kms = bw * 1e-3 / freq * G.c.value / 1e3
      print('%8.1f' % bw_kms, end=' ')

      # Channel resolution in MHz
      print(' %9.3f' % res, end=' ')

      # Channel resolution in km/s
      res_kms = res * 1e-3 / freq * G.c.value / 1e3
      print('%9.3f' % res_kms, end=' ')

      # Sensitivity per bandwidth and channel
      if ref_bw == 0:
         srms_bw_mjy   = '%8s' % 'N/A'
         srms_bw_mk    = '%8s' % 'N/A'
         srms_res_mjy  = '%8s' % 'N/A'
         srms_res_k    = '%8s' % 'N/A'
      else:
         # In mJy
         rms_bandwidth_mJy  = req_sen * np.sqrt(ref_bw / bw)
         rms_resolution_mJy = req_sen * np.sqrt(ref_bw / res)

         # In mK
         angres = data[ANG_RES][indx]
         rms_bandwidth_mK   = rms_bandwidth_mJy  / mjypermk(freq, angres, ref_freq)
         rms_resolution_K   = rms_resolution_mJy / mjypermk(freq, angres, ref_freq) / 1e3

         # Save as strings
         srms_bw_mjy  = '%8.3f' % rms_bandwidth_mJy
         srms_bw_mk   = '%8.3f' % rms_bandwidth_mK
         srms_res_mjy = '%8.3f' % rms_resolution_mJy
         srms_res_k   = '%8.3f' % rms_resolution_K
      print(' %s' % srms_bw_mjy, end=' ')
      print('%s' % srms_bw_mk, end=' ')
      print(' %s' % srms_res_mjy, end=' ')
      print('%s' % srms_res_k, end=' ')

      # Done width entry
      print('')


def printRowCorrelatorSpectralScan(data, indx, spaces=''):
   """
       Print project information for entry indx in "observations" 
       that is a spectral scan.

       Inputs:
          data   : Contains data from readObservations() in the DATA key
          indx   : Entry number in data
          spaces : Spaces to format output
   """

   # Add some space
   print('')

   # Print header
   lineExcel = getExcelFromIndex(indx)
   print('%sCorrelator information for spectral scan' % spaces)

   # Gather sensitivity information
   req_sen  = data[REQ_SENS][indx]
   ref_freq = data[REF_FREQ][indx]
   ref_bw   = data[REF_BW][indx]

   # Get spectral window information
   sps_start   = data[SPS_START][indx]
   sps_end     = data[SPS_END][indx]
   sps_bw      = getUsableBandwidth(data[SPS_BW][indx])
   sps_res     = data[SPS_SPW_RES][indx]

   # Compute quantities
   sps_res_kms = sps_res/1e3 / sps_start * G.c.value / 1e3

   # Sensitivity per bandwidth and resolution element
   if ref_bw == 0:
      srms_bw_mjy   = '%8s' % 'N/A'
      srms_bw_mk    = '%8s' % 'N/A'
      srms_res_mjy  = '%8s' % 'N/A'
      srms_res_k    = '%8s' % 'N/A'
   else:
      # In mJy
      rms_bandwidth_mJy  = req_sen * np.sqrt(ref_bw / sps_bw)
      rms_resolution_mJy = req_sen * np.sqrt(ref_bw / sps_res)

      # In mK
      angres = data[ANG_RES][indx]
      rms_bandwidth_mK   = rms_bandwidth_mJy  / mjypermk(sps_start, angres, ref_freq)
      rms_resolution_K   = rms_resolution_mJy / mjypermk(sps_start, angres, ref_freq) / 1e3

      # Save as strings
      srms_bw_mjy  = '%8.2f' % rms_bandwidth_mJy
      srms_bw_mk   = '%8.2f' % rms_bandwidth_mK
      srms_res_mjy = '%8.3f' % rms_resolution_mJy
      srms_res_k   = '%8.3f' % rms_resolution_K

   # Print spectral scale information
   print('%sStarting frequency          : %8.3f GHz' % (spaces, sps_start))
   print('%sEnding   frequency          : %8.3f GHz' % (spaces, sps_end))
   print('%sUsable bandwidth per window : %8.3f MHz' % (spaces, sps_bw))
   print('%sSpectral resolution         : %8.3f MHz' % (spaces, sps_res))
   print('%sSpectral resolution         : %8.3f km/s @ %.1f GHz' % (spaces, sps_res_kms, sps_start))
   print('%sRMS per spectral resolution : %s mJy' % (spaces, srms_res_mjy))
   print('%sRMS per spectral resolution : %s K  ' % (spaces, srms_res_k))

 
def isObsMosaic(data, indx, mtype=MOSAIC_TYPES):
   """
       Returns True or False if an observation is a mosaic of a
       type listed in MOSAIC_TYPES.

       data   : Contains data from readObservations() in the DATA key
       indx   : Row number within data
       mtype  : A list of mosaic types

       The type of mosaics are currently MOSAIC_TYPE_CUSTOM and
       MOSAIC_TYPE_RECTANGLE.
   """
   ismosaic = False
   if data[IS_MOSAIC][indx] in MOSAIC_TYPES:
      ismosaic = data[IS_MOSAIC][indx] in makeList(mtype)
   elif data[IS_MOSAIC][indx] == 'N/A' or np.isnan(data[IS_MOSAIC][indx]):
      ismosaic = False
   else:
      raise Exception('Unexpected value of IS_MOSAIC on row %d' % getExcelFromIndex(indx))

   return ismosaic


def printSummaryHeader(label, spaces=''):
   """ 
       Print header for summary table.

       Inputs:
          label  : label to print in the header row
          spaces : spaces used for formatting
   """
   print('')
   print('Summary information for %s' % label)
   print('%s%6s %6s  %-14s %-23s %12s %12s  %8s  %8s  %8s  %7s %7s %4s %4s %5s' % \
        (spaces, 'N', 'Excel', 'Project code', 'Target name', 'RA', 'Dec', 'Sky Freq', 'Ang.Res.', 'L.A.S.',
         'Polar-', 'MosArea', ' 7m?', 'TPA?', 'Spec.'))
   print('%s%6s %6s  %-14s %-23s %12s %12s  %8s  %8s  %8s  %7s %7s %4s %4s %5s' % \
        (spaces, '', 'row', '', '', 'J2000', 'J2000', '(GHz)', '(arcsec)', '(arcsec)', 'ization', 'amin^2', '', '', 'Scan?'))


def printSummarySource(number, observations, indx, spaces=''):
   """ 
       Print summary information for sources with indx of the observations.

       Inputs:
          number       : Index to keep track of number of sources
          observations : The observations data from readObservations()
          indx         : The index number if observations

       Output:
          A printed summary of the sources
   """
   project    = observations[DATA][PROJECT][indx]
   target     = observations[DATA][TARGET][indx]
   sra        = convertHmsString(float(observations[DATA][RA_DEG][indx]/15.0), ndec=1, delimiter='hms')
   sdec       = convertHmsString(float(observations[DATA][DEC_DEG][indx]), ndec=1, delimiter='dms')
#  scoords    = observations[COORDS][indx].to_string(style='hmsdms', precision=1).split()
#  sra        = scoords[0]
#  sdec       = scoords[1]
   sfreq      = '%.1f' % observations[DATA][REF_FREQ][indx]
   sangular   = '%.2f' % observations[DATA][ANG_RES][indx]
   slas       = '%.1f' % observations[DATA][LAS][indx]
   smosaic    = '-'
   saca       = '-'
   saca       = '-'
   stpa       = '-'
   sspectral  = '-'
   spol       = observations[DATA][POLARIZATION][indx].lower()
   if observations[DATA][IS_SPECTRAL][indx]:
      sspectral = 'Yes'
   if isObsMosaic(observations[DATA], indx):
      if isObsMosaic(observations[DATA], indx, mtype=MOSAIC_TYPE_RECTANGLE):
         smosaic = '%.1f' % observations[DATA][MOS_AREA][indx]
      else:
         smosaic = 'custom'
   if observations[DATA][USE_7M][indx] == True:
      saca = 'Yes'
   if observations[DATA][USE_TPA][indx] == True:
      sata = 'Yes'
   print('%s%6d %6d  %-14s %-23s %12s %12s  %8s  %8s  %8s  %7s %7s %4s %4s %5s' % \
        (spaces, number, getExcelFromIndex(indx), project, target, sra, sdec, 
         sfreq, sangular, slas, spol, smosaic, saca, stpa, sspectral))


def getMosaicCorners(ra_mosaic, dec_mosaic, width, height, angle, frame, 
                     center=None, indx=None):
   """ 
       Returns the corners of the mosaic in RA/Dec offsets from the mosaic 
       center. The function takes into account if the mosaic is specified 
       in galactic coordinates.

       Inputs
         angle      : position angle of the rectangle in degrees.
         center     : The center of the plot in RA/Dec coordinates in degrees.
                      Enter as a tuple. e.g., center=[180.0, -10.0]
         dec_mosaic : Declination center of the mosaic
         frame      : coordinate frame of the mosaic (J2000, icrs, or galactic)
         heighto    : height of the rectangle in arcseconds. Note this 
                      corresponds to "width" in the spreadsheet.
         ra_mosaic  : RA center of the mosaic
         widtho     : width of the rectangle in arcseconds. Note this 
                      corresponds to "height" in the spreadsheet.

       Output:
         A dictionary containing the RA/Dec offsets of the corners of the 
         mosaic relative to coords. If center, is none, then the mosaic 
         center is used. The dictionary keywords are UL, UR, BL, and BR.
   """

   # Get the center of the mosaic in the native frame
   if frame in ['J2000', 'icrs', 'ICRS']:
      xcen_mosaic = ra_mosaic
      ycen_mosaic = dec_mosaic
   elif frame.lower() == 'galactic':
      # Convert to ra/dec
      c = SkyCoord(ra=ra_mosaic, dec=dec_mosaic, frame='icrs', unit='degree')
      xcen_mosaic = c.galactic.l.deg
      ycen_mosaic = c.galactic.b.deg
   else:
      msg = 'Unknown mosaic frame (%s)' % (frame)
      if indx is not None:
         msg += ' on row %d' % (getExcelFromIndex(indx))
      raise Exception(msg)

   # Compute vertices relative to mosaic center
   x = 0
   y = 0
   A = -angle / 180.0 * np.pi
   results = dict()
   results['UL']  =  (x + ( width / 2. ) * np.cos(A) - ( height / 2. ) * np.sin(A),  y + ( height / 2. ) * np.cos(A)  + ( width / 2. ) * np.sin(A))
   results['UR']  =  (x - ( width / 2. ) * np.cos(A) - ( height / 2. ) * np.sin(A),  y + ( height / 2. ) * np.cos(A)  - ( width / 2. ) * np.sin(A))
   results['BL'] =   (x + ( width / 2. ) * np.cos(A) + ( height / 2. ) * np.sin(A),  y - ( height / 2. ) * np.cos(A) + ( width / 2. ) * np.sin(A))
   results['BR']  =  (x - ( width / 2. ) * np.cos(A)+ ( height / 2. ) * np.sin(A),  y - ( height / 2. ) * np.cos(A) - ( width / 2. ) * np.sin(A))

   # Set plot center.
   # The plot center is either specified in center, or if center=None,
   # the plot center is the mosaic itself.
   ra_center  = ra_mosaic
   dec_center = dec_mosaic
   if center is not None:
      ra_center  = center[0]
      dec_center = center[1]

   # Set mosaic offsets in arcseconds relative to center
   corners = dict()
   for key, c in results.items():
      # Add offsets to central coordinates in native frame of the mosaic
      xcorner = xcen_mosaic + c[0] / 3600.0 / np.cos(ycen_mosaic / 180.0 * np.pi)
      ycorner = ycen_mosaic + c[1] / 3600.0

      # Check limits on x
      if xcorner < 0:
         xcorner += 360.
      if xcorner > 360:
         xcorner -= 360.

      # Check limits on y
      if abs(ycorner) > 90.0:
         msg = 'Error setting y-axis mosaic'
         if indx is not None:
            msg += ' on row %d' % getExcelFromIndex(indx)
         raise Exception(msg)

      # Convert corner coordinates to ra/dec, if needed
      if frame.lower() == 'galactic':
         c = SkyCoord(xcorner, ycorner, frame=frame, unit='degree')
         xcorner = c.icrs.ra.deg
         ycorner = c.icrs.dec.deg

      # Compute offsets relative to mosaic center
      dra  = (xcorner - ra_center)
      ddec = (ycorner - dec_center) * 3600.0
      if abs(dra) > 180.0:
         # we are on either side of RA=0
         if dra > 0:
            dra -= 360.0
         else:
            dra += 360.0
      dalp = dra * 3600.0 * np.cos(dec_center / 180. * np.pi)

      # Save corner
      corners[key] = (dalp, ddec)

   # Done
   return corners


def plotSources(coords, observations, diameter=12,
                include=None, exclude=None,
                mosaic=False, width=60, length=60, pa=0., 
                mframe='icrs', freq=345., mosonly=False,
                plotroot='source', plotsizeo=120., plottype='pdf'):
   """
       Plot observations that are nearby the specified coordinates.

       Inputs:
          coords       : proposed coordinates from the user set by the 
                         function getSourceCoordinates()
          diameter     : diameter of the telescope in meters
          exclude      : if set, it contains a list of spreadsheet row numbers 
                         that should not be plotted.
          freq         : proposed observing frequency in GHz
          include      : if set, it contains a list of spreadsheet row 
                         numbers that can be plotted if all other criteria 
                         are set.
          length       : length of the mosaic in arcseconds
          mframe       : coordinate system of the mosaic (icrs or galactic)
          mosaic       : if True, the proposed observations are a mosaic
          mosonly      : if True, plot/print mosaic observations only
          observations : existing observations from readObservations()
          pa           : position angle of the mosaic in degrees
          plotroot     : root name for the file containing the plot. The root 
                         name will be appended with a plot number, which is
                         useful when plotting.
                         multiple sources) and the plottype.
          plotsizeo    : plot size in arcseconds
          plottype     : Type of plot to generate. The plot type must be 
                         supported by your version of python. "pdf" and "png" 
                         are usually safe. If plottype=None, then no plot is 
                         saved.
                         Default is "pdf".
          width        : width of the mosaic in arcseconds
   """
   # Set spacing for formatting purposes
   spaces = ''

   # Make a plot for each source
   for i in range(coords[ORIGINAL].size):
      # Initialize plot
      py.figure(i+1, figsize=(9,9))
      py.clf()
      ax = py.subplot(1, 1, 1, aspect='equal')

      # Set plot width
      plotsize = coords[PLOTSIZE][i]
      if plotsize is None:
         plotsize = plotsizeo

      # Find sources that overlap
      if coords[NOCOORD][i]:
         result = observations[DATA][TARGET][observations[DATA][TARGET].str.lower() == coords[ORIGINAL][i].lower()]
      else:
         # Compute separation in degrees
         sep = coords[COORDS][i].separation(observations[COORDS])
         sepArcsec = sep.arcsec 

         # Find entries within plot size
         separationArcsec = sepArcsec - 0.5*observations[DATA][MAX_SIZE]
         result = separationArcsec[separationArcsec <= (0.5*plotsize)]
      jindices = result.index

      # Print summary of observation
      if len(jindices) == 0:
         sra  = convertHmsString(float(coords[COORDS][i].ra.deg/15.0), ndec=1, delimiter='hms')
         sdec = convertHmsString(float(coords[COORDS][i].dec.deg), ndec=1, delimiter='dms')
         print('')
         print('Summary information for %s' % coords[ORIGINAL][i])
         print('    No observations found within %g x %g arcsec region centered around (ra, dec) = (%s, %s) J2000' % \
               (plotsize, plotsize, sra, sdec))
      else:
         printSummaryHeader('%g x %g arcsec region around %s' % \
               (plotsize, plotsize, coords[ORIGINAL][i]), spaces=spaces)

      # Set limits based on plotsize
      ax.set_xlim( 0.5*plotsize, -0.5*plotsize)
      ax.set_ylim(-0.5*plotsize,  0.5*plotsize)
      ax.set_xlabel('$\\Delta\\alpha\ \ \\mathrm{(arcsec)}$')
      ax.set_ylabel('$\\Delta\\delta\ \ \\mathrm{(arcsec)}$')

      # Get row lists to include/exclude
      rows_include = makeList(include)
      rows_exclude = makeList(exclude)

      # Set plot center. 
      # Sources will be plotted in offsets relative to this coordinate.
      ra_center  = coords[COORDS][i].ra.deg
      dec_center = coords[COORDS][i].dec.deg

      # Loop over observations which overlap the field
      legend_symbols = []
      legend_labels  = []
      number = 0
      for indx in jindices:
         # Get excel line
         excelRow = getExcelFromIndex(indx)
         if rows_include is not None and excelRow not in rows_include:
            continue
         if rows_exclude is not None and excelRow in rows_exclude:
            continue

         # If not mosaic and mosonly=True, then skip
         if not isObsMosaic(observations[DATA], indx) and mosonly:
            continue

         # Get ra and dec offset from nominal position in arcseconds
         if coords[NOCOORD][i]:
            dalp = 0
            ddec = 0
         else:
            # Compute offset
            ddec = (observations[DATA][DEC_DEG][indx] - dec_center) * 3600.0
            dra  = (observations[DATA][RA_DEG][indx]  - ra_center)
            if abs(dra) > 180.0:
               # we are on either side of RA=0
               if dra > 0:
                  dra -= 360.0
               else:
                  dra += 360.0
            dalp = dra * 3600.0 * np.cos(dec_center / 180.0 * np.pi)

         # Set center as offset coordinates
         xy = (dalp, ddec)

         # Print summary of observation
         number += 1
         printSummarySource(number, observations, indx, spaces=spaces)

         # Set plot color based on band
         color = getBandColor(observations[DATA][REF_FREQ][indx])

         # Plot mosaic or single pointing
         label = 'N = %d' % number
         if isObsMosaic(observations[DATA], indx, mtype=MOSAIC_TYPE_RECTANGLE):
            # Get the coordinates of the rectangle in RA/DEC offset units
            mosaicCorners = getMosaicCorners(\
                                observations[DATA][RA_DEG][indx],
                                observations[DATA][DEC_DEG][indx],
                                observations[DATA][MOS_LENGTH][indx],
                                observations[DATA][MOS_WIDTH][indx],
                                observations[DATA][MOS_PA][indx],
                                observations[DATA][MOS_COORD][indx],
                                center=[ra_center, dec_center])
            result = plotMosaic(ax, mosaicCorners, fc=color, ec=color , hatch=None, alpha=0.1)
         else:
            result = plotPrimaryBeam(ax, xy,  observations[DATA][REF_FREQ][indx], diameter,
                                     fc='None', ec=color)
         legend_symbols.append(result)
         legend_labels.append(label)

      # Loop over observations which overlap the field
#     color = getBandColor(freq)
      color = 'tan'
      if mosaic:
         corners = getMosaicCorners(ra_center, dec_center, length, width, pa, mframe)
         result = plotMosaic(ax, corners, fc=color, ec='None', alpha=0.5, linewidth=3)
      else:
         result = plotPrimaryBeam(ax, (0,0), freq, diameter, fc=color, ec='None', alpha=0.5)
      legend_symbols.append(result)
      legend_labels.append('Proposed')

      # Plot title with original entry and translation
      sra  = convertHmsString(float(coords[COORDS][i].ra.deg/15.0), ndec=1, delimiter='hms')
      sdec = convertHmsString(float(coords[COORDS][i].dec.deg), ndec=1, delimiter='dms')
      if coords[NOCOORD][i]:
         labelTranslated = ''
      else:
         labelTranslated = '%s, %s' % (sra, sdec)
      labelOriginal = '%s' % coords[ORIGINAL][i]
      fs = 14
#     ax.set_title(labelOriginal, loc='left', fontsize=fs)
#     ax.set_title(labelTranslated, loc='right', fontsize=fs)
      ax.annotate(s='Entered:', xy=(0,1.05), xycoords='axes fraction', size=fs)
      ax.annotate(s=labelOriginal, xy=(0.2,1.05), xycoords='axes fraction', size=fs)
      ax.annotate(s='Translated:', xy=(0,1.01), xycoords='axes fraction', size=fs)
      ax.annotate(s=labelTranslated, xy=(0.2,1.01), xycoords='axes fraction', size=fs)
#     py.legend(legend_symbols, legend_labels)

      # Save plot to a file
      py.show()
      if plottype is not None:
         # Set name
         root = plotroot
         if root is None:
            root = 'source'

         # Try creating plot
         try:
            plotfile = '%s%d.%s' % (root, i+1, plottype)
            py.savefig(plotfile)
            print('    Plot saved to %s' % plotfile)
         except:
            print('    Warning: Could not create plot %s.' % plotfile)
            print('    Is that plot type supported by your python extension?')


def isMovingObject(data, indx):
   """ 
       Returns True/False that object is a moving object.
   """
   return (data[RA_DEG][indx]==0 and data[DEC_DEG][index]==0)


def getCoordinatesFromName(name, data=None, lookup=True):
   """ 
       Try to get coordinates for the source name.

       First, it tries using Sesame. If that fails, then we try using
       finding the source in the observations spreadsheet.

       name   : Name of the source
       data   : Contains data from readObservations() in the DATA key
   """
   # Try resolving source name
   found = False
   if lookup:
      result = astropy.coordinates.name_resolve.get_icrs_coordinates(name)
      try:
         result = astropy.coordinates.name_resolve.get_icrs_coordinates(name)
         found = True
      except:
         pass
   if not found:
      # Initialize
      result = None

      # Is source in observation list?
      if data is not None:
         j = np.where(data[TARGET].str.lower() == name.lower())[0]
         if len(j) > 0 and not isMovingObject(data, j[0]):
            result = SkyCoord(data[RA_DEG][j[0]], data[DEC_DEG][j[0]], unit='deg', frame='icrs')

   # Done
   return result


def getSourceCoordinates(longitude, latitude, sources, inputFile, rows,
                         unit='deg', frame='icrs', data=None, 
                         spreadsheet=None, lookup=True):
   """ 
       Get list of coordinates to search.

       longitude : longitude (equatorial or galactic coordinates. Examples:
                      longitude=180.0
                      longitude='180.0d'
                      longitude='15h00m00s'
                      longitude=['10h12m13s', '180.0d', 180.0]
       latitude  : longitude (equatorial or galactic coordinates. Examples:
                      latitude=-4.0
                      latitude='-4.0d'
                      latitude=['-21d11m12s', '-4d', -4.0]
       sources   : List of source names. Examples:
                       sources='DG Tau'
                       sources='DG Tau, HL Tau, DM Tau'
                       sources=['DG Tau', 'HL Tau', 'DM Tau']
       inputFile : Name of text file containing sources to search. In addition
                   to source names, the file may contain commands to change
                   plot sizes. The pound symbol (#) is used as a comment marker.
                   Example input for the data file.
# Specify sources to sea
HL Tau
GO Tau
# Change plot size from default for remaining sources
plotsize = 480
# Additional source name
NGC7496
# Change coordinate frames and plot size
frame = galactic
plotsize = 60
# Search by coordinate
178.8590 -19.9724
       rows      : List of row numbers in excel spreadsheet
                       rows=1800
                       rows=[1800, 1850]
       unit      : Unit for longitude/latitude if they are entered as floating
                   points. Most likely values are 
                     a) unit='deg' , which means both longitude and latitude
                        are in degrees.
                     b) unit='hour,deg', which means longitude is in
                        hours and latitude is in degrees.
       frame     : Coordinate frame for longitude/latitude. Accepted values
                   are 'icrs' for equatorial or 'galactic'.
       data      : Contains data from readObservations() in the DATA key.
                   This is used to search for individual sources if 
                   a source name is not name resolved.
       lookup    : If True, resolve the source name using Sesame. 
                   lookup=False can be useful if the name listed in the 
                   spreadsheet is desired.
       spreadsheet: Name of the spreadsheet containing the observations.
                    This parameter is optional, and only used to 
                    customize the output.

       There are four option to search:
       1) Enter list of numpy array of RA/DEC.
            The format is flexible in that RA/Dec be in string, hex, or
            decimal format. If the units are not specified, the default 
            units are given by the 'unit' variable.

            For example:
               longitude=['01h05m11s', 1.32]
               latitude=['-10d05m11s', -20.0]

       2) Enter of a list of source names that will be name resolved

       3) Enter a file with coordinates/names, one entry per line
          The comment symbol per line is the pound symbol.
          The entry will name resolved first, and if that is not successful,
          it will be coordinate resolved. If that is not successful, then
          the source is skipped.

       4) Enter list of row numbers
   """

   # Initialize ra/dec list
   ra       = []
   dec      = []
   original = []
   iscoord  = []
   nocoord  = []
   plotsize = []

   # Add coordinates
   def addCoordinates(c, psize=None):
      # Save in degrees
      if c is None:
         ra.append(0)
         dec.append(0)
         nocoord.append(True)
         plotsize.append(psize)
      else:
         if type(c.icrs.ra.deg) in [list, np.ndarray]:
            zra = np.array(c.icrs.ra.deg)
         else:
            zra = np.array([c.icrs.ra.deg])
         if type(c.icrs.dec.deg) in [list, np.ndarray]:
            zdec = np.array(c.icrs.dec.deg)
         else:
            zdec = np.array([c.icrs.dec.deg])
         ra.extend(zra)
         dec.extend(zdec)
         nocoord.extend([False] * zra.size)
         plotsize.extend([psize] * zra.size)

   # Process RA/Dec
   if longitude is not None or latitude is not None:
      # Both must be set
      if longitude is None or latitude is None:
         raise Exception('Both longitude/latitude must be set if you are entering coordinates')

      # Convert entries into list so we can loop over them
      llon = makeList(longitude)
      llat = makeList(latitude)

      # Must have same size
      if llon.size != llat.size:
         raise Exception('longitude/latitude must have the same size')
      if llon.ndim != 1:
         raise Exception('Only 1D arrays can be processed for longitude')
      if llat.ndim != 1:
         raise Exception('Only 1D arrays can be processed for latitude')

      # Get coordinates
      c = SkyCoord(llon, llat, unit=unit, frame=frame)

      # Add coordinates
      addCoordinates(c)

      # Save original entries
      iscoord.extend([True] * c.icrs.ra.size)
      s = []
      for t in zip(llon, llat):
         s.append('%s, %s %s' % (t[0], t[1], frame))
      original.extend(s)

   # Process row numbers
   if rows is not None:
      # Convert to list
      lrows = getIndexFromExcel(makeList(rows))

      # Get coordinates
      c = SkyCoord(data[RA_DEG][lrows], data[DEC_DEG][lrows], unit='deg', frame='icrs')

      # Add coordinates
      addCoordinates(c)

      # Save original entries
      iscoord.extend([True] * c.icrs.ra.size)
      s = []
      for t in lrows:
         ss = 'Row %d (%s)' % (getExcelFromIndex(t), str(data[TARGET][t]))
         if spreadsheet is not None:
            ss += ' in %s' % spreadsheet
         s.append(ss)
      original.extend(s)

   # Process source names
   if sources is not None:
      # Convert into list
      names = makeList(sources)

      # Loop over names
      for name in names:
         # Get coordinates
         c = getCoordinatesFromName(name, data=data, lookup=lookup)

         # Add to coordinate list
         if c is None:
            print('Warning: Could not find coordinates for %s. Searching by object name.' % name)
         addCoordinates(c)
         iscoord.append(False)
         original.append(name)

   # Process input file
   psize_new = None
   unit_new  = unit
   frame_new = frame
   if inputFile is not None:
      # Read file one line at a time
      finp = open(inputFile, 'r')

      # Process one line at a time
      nlines = 0
      for line in finp:
         # Strip leading and trailing spaces, as well as multiple spaces
         line = re.sub(' +',' ', line.strip())

         # Remove comment portion of line
         j = line.find('#')
         if j >= 0: line = line[0:j]

         # Skip blank lines
         if line.strip() == '':
            continue

         # Is this the plot size?
         if line.lower().find('plotsize') == 0:
            value = line.split('=')[1]
            try:
               psize_new = float(value)
            except:
               pass
            continue

         # Is this the unit?
         if line.lower().find('unit') == 0:
            value = line.split('=')[1]
            try:
               unit_new = value.strip()
            except:
               pass
            continue

         # Is this the frame?
         if line.lower().find('frame') == 0:
            value = line.split('=')[1]
            try:
               frame_new = value.strip()
            except:
               pass
            continue

         # First, try if parse line as coordinates
         try:
            nlines += 1
            c = SkyCoord(line, unit=unit_new, frame=frame_new)
            addCoordinates(c, psize=psize_new)
            iscoord.extend([True] * c.icrs.ra.size)
            original.extend(['%s %s' % (line, frame_new)])
         except:
            # Coordinates was not successful. Try resolving it as a name
            c = getCoordinatesFromName(line, data=data)
            if c is not None:
               nlines += 1
               addCoordinates(c, psize=psize_new)
               iscoord.append(False)
               original.append(line)
            else:
               print('Warning: Could not find coordinates for %s. Searching by object name.' % line)
               addCoordinates(None, psize=psize_new)
               iscoord.append(False)
               original.append(line)

      # Message
      print('Read in %d sources from %s' % (nlines, inputFile))

   # Prepare results
   results = dict()
   results[COORDS]   = SkyCoord(ra, dec, unit='deg')
   results[RA]       = makeList(ra)
   results[DEC]      = makeList(dec)
   results[ORIGINAL] = makeList(original)
   results[ISCOORD]  = makeList(iscoord)
   results[NOCOORD]  = makeList(nocoord)
   results[PLOTSIZE] = makeList(plotsize)

   # Check that all elements have the same size
   keys = list(results.keys())
   for i, k in enumerate(keys):
      if len(results[k]) != len(results[keys[0]]):
         raise Exception('Arrays do not have the same size: %s and %s' % (keys[0], k))

   # Done
   return results


def row(rows, showProject=True, showCorrelator=True, 
         inputObs=LIST_OF_OBSERVATIONS, verbose=True, refresh=False):
   """ 
        Print detailed information about the project and correlator for a 
        a line in the spreadsheet.

        Inputs:
           rows          : The row numbers in the excel spreadsheet that 
                           should be printed. An integer or a list can be 
                           entered.
           showProject   : If True, then print the correlator information for 
                           each line
           showCorrelator: If True, then print the project information for 
                           each line
           inputObs      : The name of the excel spreadsheet containing the 
                           observations.
           verbose       : If True, print out messages while reading spreadsheet
           refresh       : If True, re-read the spreadsheet

        Output:
           A text listing of the project and correlator information.

        Example usage:
           import plotobs as po
           po.row(507)
           po.row([507, 508])
   """
   # Read list of previous or scheduled observations
   global OBSERVATIONS
   if OBSERVATIONS is None or refresh:
      OBSERVATIONS = readObservations(inputObs, verbose=verbose)
   observations = OBSERVATIONS

   # Loop over input excel sheet spread lines
   for excel in makeList(rows):
      # Convert to index number if observation list
      indx = getIndexFromExcel(excel)

      # Print 
      if showProject:    printRowProject(observations[DATA], indx)
      if showCorrelator: printRowCorrelator(observations[DATA], indx)


def project(projects, inputObs=LIST_OF_OBSERVATIONS, verbose=True, refresh=False):
   """ 
        Print information about a project in the spreadsheet.

        Inputs:
           projects : A string or list containing the projects to list.
           inputObs : The name of the excel spreadsheet or csv file 
                      containing the observations.
           verbose  : If True, print out messages while reading spreadsheet
           refresh  : If True, re-read the spreadsheet

        Output:
           A text listing of the project and correlator information.

        Example usage:
           import plotobs as po
           po.project('2018.1.00035.L')
           po.project(['2018.1.00035.L', '2018.1.00566.S'])
   """
   # Read list of previous or scheduled observations
   global OBSERVATIONS
   if OBSERVATIONS is None or refresh:
      OBSERVATIONS = readObservations(inputObs, verbose=verbose)
   observations = OBSERVATIONS

   # Loop over projects
   for p in makeList(projects):
      # Print header
      printSummaryHeader(p)

      # Find indices
      mask = observations[DATA][PROJECT].str.lower() == p.lower()
      indices = observations[DATA][PROJECT][mask].index

      # Print indices
      if len(indices) == 0:
         print('    No observations found for project %s' % p)
      else:
         for n, indx in enumerate(indices):
            printSummarySource(n+1, observations, indx)


def test(projects=True, rows=True, plots=True, refresh=True,
         inputObs=LIST_OF_OBSERVATIONS, verbose=True):
   """
       Run through all sources and through the projects to test the scripts
       to make all sources can be processed without run-time errors.

       projects: If True, process all projects
       rows    : If True, process all rows
       plots   : Run the plot() command on each row (very slow)
       refresh : Reload spreadsheet instead of using data in memory
   """

   # Load data
   global OBSERVATIONS
   if OBSERVATIONS is None or refresh:
      OBSERVATIONS = readObservations(inputObs, verbose=verbose)
   observations = OBSERVATIONS

   # Check projects
   if projects:
      # Get unique projects
      projectList = np.unique(observations[DATA][PROJECT])

      # Run each project
      project(projectList)

   # Check rows
   if rows:
      for indx in observations[DATA].index:
         row(getExcelFromIndex(indx))

   # Plots
   if plots:
      for indx in observations[DATA].index:
         plot(rows=getExcelFromIndex(indx))


def plot(longitude=None,    latitude=None, frame='icrs',   unit='deg',
         sources=None,      lookup=True,   inputFile=None, 
         rows=None,         include=None,  exclude=None,
         freq=345.0,        width=30.,     length=60,      pa=0., 
         mframe='icrs',     mosaic=False, 
         refresh=False,     verbose=True,  inputObs=LIST_OF_OBSERVATIONS,
         mosonly=False,
         plotroot='source', plotsize=120., plottype="pdf"):
   """ 
      Purpose:
         plot() lists and displays existing observations from Cycle 1, 2, and 
         3 and any scheduled Cycle 3 Grade A observations.

         Each observation is plotted as a single pointing with a circle the 
         size of the primary beam, or a rectangle for rectangular mosaics. For 
         custom mosaics, each individual pointing is plotted.

         One can specify an observing frequency and mosaic parameters 
         (optional), which will also be plotted with a filled circle or 
         rectangle.

      Usage:
         1) The user enters coordinates or source names, the observed 
            frequency, the proposed mosaic parameters, and the size of the 
            plot region (PLOTSIZE)

         2) plot() will list all observation within PLOTSIZE size, as well
            as plot the observations.

      How to input source name or coordinates:
         Parameters:
            exclude   : If set, it contains a list of spreadsheet row numbers 
                        that should not be plotted.
            frame     : reference frame for longitude, either "icrs"
                        (i.e., equatorial) or "galactic"
            include   : If set, it contains a list of spreadsheet row numbers 
                        that can be plotted if all other criteria are set
            inputFile : Input list sources or positions
            inputObs  : Input excel file containing previous observations 
                        and scheduled observations
            latitude  : Latitude coordinate with reference frame specified 
                        by "frame". Examples:
                           latitude=-20.0, unit='deg'
                           latitude='-20 deg'
                           latitude=['-20 deg', '20d00m00s']
            longitude : Longitude coordinate with reference frame specified 
                        by "frame". Examples:
                           longitude=100.0, unit='deg'
                           longitude='100 deg'
                           longitude=['100 deg', '17h15m:10s']
            lookup    : If True, resolve the source name using Sesame. 
                        lookup=False can be useful if the name listed in the 
                        spreadsheet is desired.
            mosonly   : If True, print/display only mosaic observations
            refresh   : If True, re-read excel spreadsheet into memory
            rows      : List of row numbers in excel spreadsheet
            sources   : List of source names. The coordinates of the source 
                        names are obtain from the Sesame database, and if the 
                        name is not resolved, the name is searched 
                        (case-insensitive) in the input spreadsheet. If 
                        lookup=False, the search in the Sesame database is 
                        skipped.
            unit      : units of longitude and latitude if not specified by
                        in longitude/latitude. Examples:
                           unit='deg' implies longitude/latitude are in degrees.

                           unit='hour,deg' implies longitude is in hours 
                           and latitude is in degrees.
            verbose   : If True, print out messages when reading the excel 
                        spreadsheet

         Plot parameters
            plotroot  : Root name for the file containing the plot. The root 
                        name will be appended with a plot number (useful when 
                        plotting multiple sources) and the plottype.
            plotsize  : Size of the square region to plot in arcseconds
            plottype  : Type of plot to generate. The plot type must be 
                        supported by your version of python. "pdf" and "png" 
                        are usually safe. Default is "pdf". Set plottype=None 
                        to not save the plots to a file.

         Proposed observed parameters
            freq      : Proposed observing frequency in GHz
            length    : Proposed length of the mosaic in arcseconds
            mframe    : Coordinate system of the mosaic (icrs or galactic)
            mosaic    : If True, the proposed observations are a mosaic
            pa        : Proposed position angle of the mosaic in degrees
            width     : Proposed width  of the mosaic in arcseconds

         Usage:
            There are three options to search for source names:
               1) Enter list of coordinates. 
                  The format is flexible in that coordinates may be a string, 
                  hex, or decimal format. If the units are not specified, the 
                  default units are given by the 'unit' variable.

                  For example:
                     longitude=['01h05m11s', 1.32]
                     latitude=['-10d05m11s', -20.0]

               2) Enter of a list of source names that will be name resolved 
                  by querying the Sesame database. An internet connection 
                  must be available for the coordinate lookup

               3) Enter a file with coordinates/names, one entry per line
                  The comment symbol per line is the pound symbol.
                  The entry will be name resolved first, and if that is not 
                  successful, it will be coordinate resolved. If that is not 
                  successful, then the source is skipped.

               4) The source name will be matched to the OBSERVATION list.
                  The search will be case insensitive.

         Examples: See Header of the package for Examples.
   """
   # If longitude is set but no other source parameters are set, then
   # assume the following:
   # 1) If longitude is a string, then assume it is a source name
   # 2) If it is an integer, then it is a row number
   if longitude is not None and latitude is None and \
      rows is None and sources is None and inputFile is None:
      if str in [type(longitude)]:
         sources = longitude
         longitude = None
      elif int in [type(longitude)]:
         rows = longitude
         longitude = None

   # Check inputs
   error = False
   # Longitude/latitude must both be set or not at all
   if (longitude is not None and latitude is     None) or \
      (longitude is     None and latitude is not None):
      error = True
      print('Error: Longitude and latitude must both be set to enter coordinates')
   # Some coordinates must be set
   if (longitude is None or latitude is None) and \
      rows is None and \
      inputFile is None and \
      sources is None:
      error = True
      print('Error: longitude/latitude, sources, rows, or inputFile must be set')
   # Check frame of coordinates
   if frame.lower() not in ['icrs', 'galactic']:
      error = True
      print('Error: frame must be set to "icrs" or "galactic"')
   # Check plotsize
   if plotsize <= 0:
      error = True
      print('Error: plotsize must be > 0')
   # Check frequency
   if freq <= 0:
      error = True
      print('Error: frequency must be > 0')
   # Check mosaic parameters, if mosaic is set
   if mosaic:
      if length <= 0:
         error = True
         print('Error: length must be > 0')
      if width <= 0:
         error = True
         print('Error: width must be > 0')
      if mframe.lower() not in ['icrs', 'galactic']:
         error = True
         print('Error: mframe must be set to "icrs" or "galactic"')
   if error:
      print('')
      print('Error starting plotobs_cycle7_supp')
      return

   # Read list of previous or scheduled observations, if not entered 
   # on command line
   global OBSERVATIONS
   if OBSERVATIONS is None or refresh:
      OBSERVATIONS = readObservations(input=inputObs, verbose=verbose)
   observations = OBSERVATIONS

   # Get coordinates of source(s)
   coords = getSourceCoordinates(longitude, latitude, sources, inputFile, rows,
                          unit=unit, frame=frame, data=observations[DATA], 
                          spreadsheet=inputObs, lookup=lookup)

   # Plot results
   plotSources(coords, observations,
               include=include, exclude=exclude,
               freq=freq, mosonly=mosonly,
               mosaic=mosaic, mframe=mframe, width=width, length=length, pa=pa,
               plotroot=plotroot, plottype=plottype, plotsizeo=plotsize)
