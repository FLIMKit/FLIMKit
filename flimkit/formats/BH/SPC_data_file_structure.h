#pragma pack(push,1)


/* This file contains defines and structure definitions which can be useful
        in case of writing a program to analyze .set, .sdt and .spc files
        created using BH software
    valid for SPCM v.9.66 and newer
*/

/*

/////////////////////////////////////////////////////////////////////////
////////////         .set  and .sdt files structure   ///////////////////
/////////////////////////////////////////////////////////////////////////

The setup files ( extension .set ) consist of:
  1 - a file header containing structural data which are used
        to find the other parts of the file
  2 - the 'file information' which was typed in when the file was saved
  3 - the system setup data for hardware and software



The data files ( extension .sdt ) consist of:
  1 - a file header containing structural data which are used
        to find the other parts of the file
  2 - the 'file information' which was typed in when the file was saved
  3 - the system setup data for hardware and software
  4 - one or more measurement description blocks which contain
       the system parameters corresponding to the particular data blocks
  5 - data blocks containing a set of curves from one measurement each,
       along with information to which measurement description block
       they correspond

*/

#define BH_HDR_LENGTH         42
#define BH_HEADER_CHKSUM      0x55aa
#define BH_HEADER_NOT_VALID   0x1111
#define BH_HEADER_VALID       0x5555
    // different header chksum for files created with SPCM v.992 and higher with LZ4 compressed blocks
#define BH_HEADER_CHKSUM_992  0xaa55

// all offsets are given from the beginning of the file
// all parts lengths are given in bytes

#define REV_WITH_APP_TYPE  12 // bin_identifier on bits 11-4
#define REV_12B_APP_TYPE   14 // bin_identifier on bits 15-4
#define REV_2GB_BLOCKS     15 // huge data blocks >128MB <= 2GB
#define SOFTWARE_REV  15   // current software revision

 //  1 - File header
typedef struct  {
   short    revision;  // software revision & module identification
                       //   lowest bits 0-3 -   software revision ( >= 12(decimal))
                       //        current value = 15 - support for huge data blocks >128MB <= 2GB
                       //   bits 11-4   - BH module type
                       //      meaning of this field values (hex):
                       //        0x20 -SPC-130, 0x21 - SPC-600, 0x22 - SPC-630,
                       //        0x23 -SPC-700, 0x24 - SPC-730, 0x25 - SPC-830,
                       //        0x26 -SPC-140, 0x27 - SPC-930, 0x28 - SPC-150,
                       //        0x29 -DPC-230, 0x2a - SPC-130EM, 0x2b -SPC-160,
                       //        0x2e -SPC-150N, 0x80 - SPC-150NX, 0x81 -SPC-160X,
                       //        0x82 -SPC-160PCIE, 0x83 -SPC-130EMN, 0x84 -SPC-180N,
                       //        0x85 -SPC-180NX,   0x86 -SPC-180NXX, 0x87 -SPC-180N-USB,
                       //        0x88 -SPC-130IN,   0x89 -SPC-130INX, 0x8a -SPC-130INXX
                       //        0x8b -SPC-QC-104,  0x8c -SPC-QC-004, 0x8d -SPC-QC-106,
                       //        0x8e -SPC-QC-006
                       //
                       //   highest bits 15-12 - module subtype -
                       //         1 for SPC-150NX  =  SPC-150NX-12 ( 12.5ns TAC range)
   long     info_offs; // offset of the info part which contains general text
                       //   information (Title, date, time, contents etc.)
   short    info_length;  // length of the info part
   long     setup_offs;   // offset of the setup text data
               // (system parameters, display parameters, trace parameters etc., binary setup)
   unsigned short    setup_length;  // length of the setup data
   long     data_block_offs;   // offset of the first data block
   short    no_of_data_blocks; // no_of_data_blocks valid only when in 0 .. 0x7ffe range,
                               // if equal to 0x7fff  the  field 'reserved1' contains
                               //     valid no_of_data_blocks
   unsigned long     data_block_length;     // length of the longest block in the file
                                            //        ( not compressed ) in bytes
   long     meas_desc_block_offs;  // offset to 1st. measurement description block
                                   //   (system parameters connected to data blocks)
   short    no_of_meas_desc_blocks;  // number of measurement description blocks
   short    meas_desc_block_length;  // length of the measurement description blocks
   unsigned short    header_valid;   // valid: 0x5555, not valid: 0x1111
   unsigned long     reserved1;      // reserved1 now contains no_of_data_blocks
     // reserved2 now contains length (in int words) of data block extension, normally 0,
     //   data block extension contains info data for histograms data blocks
   unsigned short    reserved2;
   unsigned short    chksum;            // checksum of file header
   }bhfile_header;


/*
  2 - File Info
This part contains the general information which has been typed in when the data was saved.
The info part is stored in ASCII. An example is given below.
The part must be present in the file otherwise SPCM software cannot recognize the file type
  and finally cannot load the file

*IDENTIFICATION
  ID  : _SPC Setup & Data File_
  Title : startup
  Version : 1 810 M
  Revision    : 12 bits ADC
  Date  : 10-10-1997
  Time  : 12:29:01
  Author  : Bond, James
  Company : Unknown
  Contents    : Dye sample from Dr. No
*END

SPCM software looks for "*IDENTIFICATION" string and then looks for one of listed
  below possible identifiers
ID line is tested by the application to check whether the file is a valid SPC file
Possible identifiers are listed below
If the user creats his own data file ( w/o setup) ID should be set to
  dll_data_identifier string

Version line contains File type and software revision used to create the file
  File type = 1 - for files created in non-Fifo measurement modes
            = 2 - for files created in Fifo measurement mode
            = 3 - for files created in FifoImage measurement mode
Revision line contains ADC resolution
*/

// for .set files ( setup only )
const char setup_identifier[32] =      "SPC Setup Script File";

// for normal .sdt files ( setup + data )
const char  data_identifier[32] =      "SPC Setup & Data File";

// for .sdt files created automatically in Continuous Flow mode measurement
//     ( no setup, only data )
const char  flow_data_identifier[32] = "SPC Flow Data File";

// for .sdt files created using DLL function SPC_save_data_to_sdtfile
//     ( no setup, only data )
const char  dll_data_identifier[32]  = "SPC DLL Data File";

// for .sdt files created in Fifo & FifoImage modes
//     ( setup, data blocks in Fifo mode = Decay, FCS, FIDA, FILDA, MCS, MCS_TA curves
//                                          for each used routing channel
//              data blocks in Fifo Image mode = IMG, IMG_MCS, IMG_INT
//     )
const char  fcs_data_identifier[32]  = "SPC FCS Data File";



/*
  3 - Setup

The setup block contains all the system parameters, display parameters, trace parameters etc
  stored in ASCII.
From the software version 8.1  setup block contains also binary part.

It is used to set the SPC system (hardware and software) into the same state as
  it was in the moment when the data file was stored.
The values are stored together with an identifier of the particular parameter.
This method allows to maintain compatibility between different SPC versions.
If a parameter is missing in the setup part, i.e.
  if a file from an older software version is loaded, a default value is used
   when the file is loaded.
For Multi SPC Systems the system parameters section contains subsections
   for module parameters which are separate for the individual modules.

ASCII part of the setup block starts with the line
*SETUP
and ends with the line
*END

 DI_ parameters in ASCII part of the setup are now obsolete and are present
 for compatibility reason only - parameters from binary setup are used

After ASCII part follows the binary part of the setup  ( for the files created
    with the software version 8.1 and next )

It starts with the text BIN_PARA_BEGIN:
Directly after BIN_PARA_BEGIN: text comes:
  -  1 byte = '0' which is EOS character ( End Of String )
  -  4-byte value which is equal to the length of binary setup
          ( all suqsequent headers and structures up to *END text )
  -  BHBinHdr bh_bin_hdr
  -  SPCBinHdr spc_bin_hdr
  -  structures defined in spc_bin_hdr header

binary setup ends with the line
*END

binary setup contains mainly display parameters for different graph windows
 ( Main graph (2D, 3D), FCS and other (in future) ),
 which are used by the main software to restore display parameters
   of the specific windows
 It contains also parameters specific for external devices controlled by SPCM software
   like GVD-120 module, SyncDelay Box or parameters for Zscan with Zeiss AxioObserver

In order to reach binary part of the setup user should read the whole setup part
  to the buffer and then read strings from the buffer
    until the string BIN_PARA_BEGIN: is found.



*/

#define MAX_SPC     4    // max no of SPC modules accepted by SPCM application


typedef struct {
   unsigned long     soft_rev;       // software revision
   unsigned long     para_length;    // currently set to 0
   float             file_rev;       // file revision = SPC_FILE_VER ( 2.0 )
   unsigned short    reserved2;
   }BHBinHdr;


// all offsets in SPCBinHdr and SPCBinHdrExt are given from the beginning of
//            BHBinHdr structure  ( just after 4-byte length of binary setup )
// all parts lengths are given in bytes

typedef struct {
  unsigned long     FCS_old_offs;    // offset of FCSParam structure for FCS parameters
  unsigned long     FCS_old_size;    //  FCSParam structure size
                    //  FCSParam is obsolete - needed only for compatibility
                    //  with software v. 8.1 - 8.2
                    //  from v.8.3 FCS parameters are included further in HSTParam
  unsigned long     gr1_offs;        // offset of GR_Param structure for main graph window
  unsigned long     gr1_size;        //  GR_Param structure size
  unsigned long     FCS_offs;        // offset of HSTParam structure for FCS parameters
  unsigned long     FCS_size;        //  HSTParam structure size
  unsigned long     FIDA_offs;       // offset of HSTParam structure for FIDA window parameters
  unsigned long     FIDA_size;       //  HSTParam structure size
  unsigned long     FILDA_offs;      // offset of HSTParam structure for FILDA window parameters
  unsigned long     FILDA_size;      //  HSTParam structure size
  unsigned long     gr2_offs;        // offset of GR_Param structure for the 1st additional graph
  unsigned short    gr_no;           // number of additional GR_Param structures, ( size = gr1_size )
  unsigned short    hst_no;          // number of additional HSTParam structures, ( size = FCS_size )
                                     //   = MAX_NO_OF_HST - 3 - currently = 2 for MCS and MCS_TA histograms
  unsigned long     hst_offs;        // offset of the 1st additional HSTParam structure
  unsigned long     GVD_offs;        // offset of GVDParam structure for GVD parameters
  unsigned short    GVD_size;        //  GVDParam structure size
  unsigned short    FIT_offs;        // offset of FITSetup structure for FIT parameters
  unsigned short    FIT_size;        //  FITSetup structure size
  unsigned short    extdev_offs;     // offset of external devices setup structures ( ExtDevSetup)
  unsigned short    extdev_size;     //   size of external devices setup structures ( ExtDevSetup)
  unsigned long     binhdrext_offs;  // offset of SPCBinHdrExt - extension of SPCBinHdr
  unsigned short    binhdrext_size;  //   size of SPCBinHdrExt - extension of SPCBinHdr
  }SPCBinHdr;

typedef struct {     // extension of SPCBinHdr
  unsigned long   MCS_img_offs;     // offset of MIMGParam structure for MCS Image parameters
  unsigned long   MCS_img_size;     //  MIMGParam structure size
  unsigned short  mom_no;           // number of MOMParam structures
  unsigned short  MOM_size;         //  MOMParam structure size
  unsigned long   mom_offs;         // offset of the 1st MOMParam structure
  unsigned long   syspar_ext_offs;  // offset of SysParExt structure for extension of system parameters
  unsigned long   syspar_ext_size;  //  SysParExt structure size
  unsigned long   mosaic_offs;      // offset of MosaicParam structure for extension of system parameters
  unsigned long   mosaic_size;      //  MosaicParam structure size
  unsigned long   WF_img_offs;      // offset of WFParam structure for Wide-field Image parameters
  unsigned short  WF_img_size;      //  WFParam structure size
  unsigned long   WndLayout_offs;   // offset of WndLayout structure for Windows Layout parameters
  unsigned short  WndLayout_size;   //  WndLayout structure size
  unsigned long   trpar_ext_offs;   // offset of TrParExt structure for extension of 3D Traces parameters
  unsigned short  trpar_ext_size;   //  TrParExt structure size
  unsigned long   CorPar_offs;      // offset of 1st CorPar structure for multichannel detector correction
  unsigned short  CorPar_size;      //  CorPar structure size
  unsigned short  CorPar_number;    //  number of CorPar structures - max 4 ( 1 per SPC module)
                                    //      bit 8 = 1 - Detector Correction Enabled
  unsigned long   LifeTrPar_offs;   // offset of 1st LifeTrPar structure for 3D Lifetime traces
  unsigned short  LifeTrPar_size;      //  LifeTrPar structure size
  unsigned short  LifeTrPar_number;    //  number of LifeTrPar structures - max 8 ( 1 per 3D trace)

  unsigned long   HST3DPar_offs;    // offset of 1st HST3DParam structure for 3D histogram traces
  unsigned short  HST3DPar_size;    //  HST3DParam structure size
  unsigned short  HST3DPar_number;  //  number of HST3DParam structures - MAX_NO_OF_3D_HST

  unsigned long   dcc_ext_offs;   // offset of DccParExt structure for extension of DCC parameters
  unsigned short  dcc_ext_size;   //  DccParExt structure size

  unsigned char   extension[162];   //  for future use, keep the size of SPCBinHdrExt = 240B
  }SPCBinHdrExt;


/*
SPCBinHdr header includes offsets of GR_Param structure for the main display window
  and HSTParam structures for FCS, FIDA and FILDA windows.
Another GR_Param structures can be included if additional 3D display windows were
  opened
Another HSTParam structures can be included if additional histograms windows were
  opened ( e.g. MCS (Macro Time) histogram )
GVDParam structure contains parameters of GVD-120 module which controls BH Galvano Scanner.
FITParam structure contains start values of FCS fitting parameters
Other fields will be used in future software versions
FCS, FIDA, FILDA, MCS windows parameters are defined below in HSTParam structure
The structure contains attributes of histogram display windows,
  which can be visible in FIFO mode
*/

typedef struct {
  int   top;
  int   left;
  int   height;
  int   width;
  int   flags;  // 0x1 - visible, bits 4-5 = floating : 0 - VAL_FLOAT_NEVER,
                //              1 - VAL_FLOAT_APP_ACTIVE, 3 -  VAL_FLOAT_ALWAYS
                // bit 8 - 0x100 - scale panel's contents on resize
                // bit 9-11 - on bits 16-27 ( 12b ) special value of :
                //                         1 - top, 2 - left, 3 - height, 4 - width
  }BHPanelAttr;



typedef struct {
  int   status;
  char  name[20];
  BHPanelAttr data;
  }BHPanelData;


typedef union
{
  int             i;
  unsigned int    u;
  int             b;
  short           h;
  unsigned short  g;
  long            l;
  unsigned long   m;
  float           f;
  double          d;
  char            c;
} DparVal;





typedef struct
{
  DparVal maxcnt;     //  DI_MAXCNT
  DparVal lbline;     //  DI_LBLINE
  DparVal bline;      //  DI_BLINE
  int  scale;         //  DI_SCALE
  int  grid;          //  DI_GRID  X : bit1, bit3 minor/ Y : bit2, bit 4 minor/ bit0 = X+Y
  int  plot_style;    //  DI_TRACE
  int  point_style;   //  DI_PSTYLE
  int  line_style;    //  set to 0
  int  point_freq;    //  DI_PFREQ
  int  resize;        //  0 - not set, 1- Full size, 2 - Proportional, 3- other
  int   grid_color;   //  DI_GCOL_F
  int   bkg_color;    //  DI_GCOL_B
  int   d2_mode;      //  DI_2DDIS
  short reverse;      //  DI_XREV - bit 0, DI_YREV - bit 1
  short ascale;       //  autoscale for 2D
  DparVal xlow;
  DparVal xhigh;
  int  xscale;        //  Xscale log = 1
  DparVal yhilim;     // high limit for y scale
} Disp2DAttr;


#define MAX_NO_OF_INTERVALS  8

#define MAX_WIN 5   // w/o  MTWIN

#define XWIN    0
#define YWIN    1
#define TWIN    2
#define SXWIN   3
#define SYWIN   4
#define MTWIN   5  // time windows for MCS images
#define MMWIN   6  // time gate windows for Moments mode

typedef struct _Interval {
    unsigned int    from;
    unsigned int    to;
    }Interval;

typedef struct _Window {
    short    no_of_intervals;
    short    equidistant;
    unsigned int    end_point;
    int      no_pid;
    int      eq_pid;
    Interval interval[MAX_NO_OF_INTERVALS];
    } Window;




typedef struct  {    // obsolete, needed only for compatibility with software v. 8.1 - 8.2
  int      view_decay;
  int      view_fcs;
  float    cortime; // correlation time [ms]
  BHPanelAttr  pnl_attr;
  Disp2DAttr   d2para;
  }FCSParam;


// Histogram traces parameters ( FCS, FIDA, FILDA, MCS )
typedef struct  {
  short    active;         // (0)1 - (not) active
  short    module;
  short    curve;          // routing channel no
  short    cross_curve;    // cross correlation channel number for FCS
                      //    ( when = curve and module = cross_module -  auto FCS )
  short    cross_module;   // cross correlation module number for FCS
  int      color;          // trace color
  int      points;         // number of points in trace
  }HSTtrace;  //



#define MAX_NO_OF_TRACES  16

#define MAX_NO_OF_HST 5     // max number of histogram windows ( number of HSTParam structures)

typedef struct{        // structure for FCS, FIDA, FILDA MT, MCS, MCS_TA histogram windows
  int               active;   // window visible, histogram is calculated
  unsigned short    hst_type; // additional histogram parameters
                       // for FCS:  0 - linear FCS with log binning ( 100 bins/log )
                                 // when bit 15 = 1 ( 0x8000 ) - Multi-Tau FCS
                                 //           where bits 14-0 = ktau parameter
                       // for MCS_TA:
                              // bit 0 = 1  - use 'invalid' photons
                              // bit 1-2  =  marker no used as trigger
                              // bit 3-5  =  1..7   4 ** value = points number
                              // bit 8-15 =  time per point in MT clock units  (highest 8 bits )
  unsigned short    par_ext;  // parameter extension
                              //   for MCS_TA:  time per point in MT clock units  (lower 16 bits )
  float             time;     // in [ms], correlation time for FCS,
                              // period time for FIDA, FILDA, time per point for MT, MCS, MCS_TA
  BHPanelAttr       pnl_attr;
  Disp2DAttr        d2para;
  HSTtrace          htrace[MAX_NO_OF_TRACES];
                    // extra parameters
  int               ipar_ext1;
  int               ipar_ext2;
  float             fpar_ext;  // for MCS_TA: offset in percent  -100 .. +100
  }HSTParam;



// definitions for GVD-120/140  module


typedef struct {          /* structure for GVD module data  */

  short active;         // most of the library functions are executed
                        //   only when module is active ( not 0 ), default 0
  unsigned short frame_size;
                        // bits 0-3  - lg of frame size ( quadratic frame)
                        //   values in range: 4 (16x16) ... 12 (4096x4096),
                        //            ( default 9, 512x512)
                        // bits 4-7  - lg of frame size X ( rectangular frame)
                        //   values in range: 4 (16) ... 12 (4096),
                        //            ( default 9, 512)
                        // bits 8-11 - lg of frame size Y ( rectangular frame)
                        //   values in range: 4 (16) ... 14 (16384),
                        //            ( default 9, 512)
                        // bit 15 = 0 - quadratic frame, = 1 - rectangular frame

  short lasers_active;  // bit 0(8) - laser 0(1) active, default 0,  for GVD-120 module
                        // bit 1    - laser 0 & 1 off during flyback , default 0
                        // bit 2    -  = 0 - laser 0 is the first laser when multiplexing lasers
                        //             = 1 - laser 1 is the first laser when multiplexing lasers ,
                        //              default 0, bit 2 valid for FPGA version > B2
                        //  for GVD-140
                        //  bit 0(8) - laser 0(1) active, default 0,
                        //  bit 1    - lasers 0-3 off during flyback , default 0
                        //  bit 2 is not used, instead lasers_active1-4 is used
                        //  bits 9,10 - lasers 2,3 active, default 0
                        //  when multiplexing is on, additionally lasers_active1-4 defines
                        //              which laser and in each phase is active
                        //  bit 12 - DCS-Box in use (1) or not ( 0 ), default 0
                        //  either DCS_BOX or Attenuator Down of Laser 4  can be used ( they use the same pin )
  unsigned short multiplex;    // lasers multiplexing mode ( bits 0-1): 0 - off,
                        //           1 - within each pixel, 2 - after each line,
                        //           3 - after each frame , default 0
                        //  bits 2-15 - length of first laser phase in 1/100th of percent
                        //                when mode 1 ( pixel multiplexing), default 5000 ( 50%)
                        // for GVD-120 :  ( first laser = laser 0 ( FPGA version < B3 )
                        //                              = bit 2 of lasers_active  ( FPGA version > B2 )
                        // for GVD-140 : lasers_active1-4 defines sequence of switching lasers

  short limit_scan;     // 0 - unlimited scan ( default ), 1 - scan specified no of frames
                        // 2 - repeated scan of specified no of frames ( used with trigger)
  unsigned short frame_counter;  // no of frames to scan when limit_scan > 0, default = 1
  unsigned short scan_polarity;  // default 0 ( all active low )
                        //   bit 0 - pixel clock polarity, bit 1 - polarity of HSYNC (Line),
                        //   bit 2 - polarity of VSYNC (Frame)
                        //          bit = 0 - falling edge(active low)
                        //          bit = 1 - rising  edge(active high)

  short scan_type;      // bit 0  = 0 scan frame ( default ), = 1 - scan one line only
                        // bit 3-1  line scanning type
                        //    = 0 ramp continuous ( default ) ( scans also during the pixel time ),
                        //    = 1 ramp steps (same position during the pixel) - increases min line time
                        //    = 2 sinus continuous,
                        // bit 4 = 0 normal scan modes (default), = 1 point mode (for GVD-140 from firmware v. A5)

  float line_time;      // line time in sec
  float zoom_factor;    // zoom factor for quadratic frame 1( default ) .. 1024,
                        //     max scan range ( amplitude)  is divided by the factor
                        //     max depends on frame size: 4 for frames 2048x2048,
                        //          8 for frames 1024x1024, 16 for frames 512x512
                        //          32 for frames smaller than 512x512
  float offset_x;       // offset X of the centre of zoomed area from the centre X
                        //     of the total scan range, -100 % .. 100 % ( default 0 )
  float offset_y;       // offset Y of the centre of zoomed area from the centre Y
                        //     of the total scan range, -100 % .. 100 % ( default 0 )
  float park_offs_x;    // offset X of the park position from the centre X
                        //     of the total scan range, -100 % .. 100 % ( default 0 )
  float park_offs_y;    // offset Y of the park position from the centre Y
                        //     of the total scan range, -100 % .. 100 % ( default 0 )
  float l1_power;       //  Laser 1 power control 0 - 100%, default 50
  float l2_power;       //  Laser 2 power control 0 - 100%, default 50
  float  rect_zoom_x;   //  X zoom factor for rectangular frame, 1( default ) .. 1024,
  float  rect_zoom_y;   //  Y zoom factor for rectangular frame, 1( default ) .. 1024,
  short scan_rate;      //  0 - auto,scan rate fastest possible ( default ),
                        //  1 - scan rate defined by line_time
  short park_center;    //  bit 0 = 1 when parked, set beam position in the centre
                        //            of scanned area, default 0
                        //  bit 4  ( read only ) shows current state of beam parking
                        //           1 - beam is parked, 0 - beam is not parked
  short scan_trigger;   //  External trigger condition ( FPGA version > B0 )
                        //           bits 1 & 0 mean :
                        //             00 - ( value 0 ) external trigger disabled (default),
                        //             01 - ( value 1 ) external trigger active low,
                        //             10 - ( value 2 ) external trigger active high
  unsigned short  control2;           //  for GVD-140, for internal use, default 0
  float l3_power;       //  for GVD-140, Laser 3 power control 0 - 100%, default 50
  float l4_power;       //  for GVD-140, Laser 4 power control 0 - 100%, default 50
  unsigned short  multiplex2;  // for GVD-140
                        //  bits 0-13 - start of the second laser phase in 1/100th of percent
                        //        when pixel multiplexing and 4 lasers sequence is used, default 0 ( 0%)
                        //  lasers_active1-4 defines sequence of switching lasers
  unsigned short  multiplex3;  // for GVD-140
                        //  bits 0-13 - start of the third laser phase in 1/100th of percent
                        //        when pixel multiplexing and 4 lasers sequence is used, default 0 ( 0%)
                        //  lasers_active1-4 defines sequence of switching lasers

  unsigned short  lasers_active1_4;   //  for GVD-140, default 0
                        //  defines which laser is active in subsequent multiplexing phases
                        //  bits 0-3   Laser 1 activity in phases 1-4
                        //  bits 4-7   Laser 2 activity in phases 1-4
                        //  bits 8-11  Laser 3 activity in phases 1-4
                        //  bits 12-15 Laser 4 activity in phases 1-4
                        //  if multiplexing is off, laser is active when minimum one of its 4 bits is set to 1
                        //   always a proper bit in lasers_active must be set to really activate specific laser

         // all fields below are valid for GVD-140 ( from firmware v. A5 ) in Point Mode
         //    ( bit 4 of scan_type = 1 )
   unsigned short  point_type;    // lasers multiplexing options in GVD-140 point mode
                                  //   bits 2-0 - multiplexing mode, default 0
                                  //   bit  0 = 0 multiplexing off, = 1 on, default 0
                                  //   bit  1 = 0 NO plim mode (default), = 1 plim mode
                                  //   bit  2 = 0 single point mode (default), = 1 multi point mode
                                  //   bits 2-1  = 00 - SP-FL SinglePoint Flim ( default )
                                  //             = 01 - SP-PL SinglePoint Plim
                                  //             = 10 - MP-FL MultiPoint Flim
                                  //             = 11 - MP-PL MultiPoint Plim
                                  // lasers multiplexing mode in not point_mode are defined in structure member 'multiplex'
  float    point_time_resol;      //  point time resolution[s] in the range 100ns - 10ms, default 100ns
  float    point_time;            //  point time [s] in the range 200ns - 10s, default 10µs
  //   pt_multiplex1-4 -  bits 0-13 - switch point (end) of the ON phase for 4 lasers in 1/100th of percent
                            //       in GVD-140 Point mode , default 10000 ( 100% )
                            //  lasers_active1-4 defines sequence of used lasers
  unsigned short  pt_multiplex1;      //  laser 1
  unsigned short  pt_multiplex2;      //  laser 2
  unsigned short  pt_multiplex3;      //  laser 3
  unsigned short  pt_multiplex4;      //  laser 4
  unsigned short  control3;           //  for GVD-140, for internal use (reg 0x2A), default 0
  char     reserve[62];    // keep always total size = 160 B
  }GVDdata;




typedef struct{        // structure for GVD control data
  int               active;   // GVD in use
  unsigned int      dcs_and_beam_blank;    // bit 0 = beam_blank, bits 1-31 DCS-BOX parameters
  GVDdata           gvd_data;
  unsigned short    DAC_per_step;    //  DAC_per_step 1 .. 1024,  bits 15-12 options
                    //   bit 12 = 1 - automatically synchronize image size in system parameters
  short             line_pulse_shift;
  BHPanelAttr       pnl_attr;
        // all fields below are valid for GVD-140 ( from firmware v. A5 ) in Point Mode
  short          point_time_unit;   // 0 - ns, 1 - µs, 2 - ms
  short          lock_time;         // 0 - Lock percentage value of Laser's OnTime when changing point_time
                                    // 1 - Lock  time value of Laser's OnTime when changing point_time
  unsigned short  MP_PL_multiplex[4];    //  last multiplex values in mode MP_PL for 4 lasers
  unsigned short  SP_FL_multiplex[4];    //  last multiplex values in mode SP_FL for 4 lasers
  unsigned short  SP_PL_multiplex;       //  last multiplex values in mode SP_PL ( one value for all lasers)
  char            reserve[42];   // keep always total size = 256 B
  }GVDParam;



// FCS fitting parameters


typedef struct{           // structure for FCS fitting parameters
  double      F;          //  triplet fraction [%]
  double      N;          //  avg. number of molecules
  double      tau_trip;   //  triplet decay time [µs]
  double      y_1;        //  fraction of 1st component [%]
  double      tau_1;      //  diffusion time (1st comp.) [µs]
  double      tau_2;      //  diffusion time (2nd comp.) [µs]
  double      s;          //  structure parameter
  double      chi_square; //  Indicator for goodness of fit
  }FITPar;

#define    FPAR_NO    8   // number of fields of FITPar
                          //     number of fitting parameters + 1

typedef struct{           // structure for FCS fitting results
  FITPar      curr_fit;   // current values  of fitting parameters
  short       st_no;      // fitting curve start point no
  int         iter_cnt;   // total iteration counter
  short       last_err;   // last return value
  }FITresult;

typedef struct{        // structure for FCS fitting parameters
  int         fit_on;   // online fitting active
  short       fit_trace;//  bitwise value shows which traces are fitted
  short       fix;      //  bitwise value shows which fields of FITPar are fixed
                        //       during fitting
  double      st_time;  // start time of fitting [µs]
  FITPar      st_fit;   // start values ( initial guess) of fitting parameters
  int         reserve1;
  double      reserve2;
  }FITSetup;



// Z scan  Axio microscopes

typedef struct{        // structure for Z scan  Axio microscopes
  short              connected;  // observer_mode =  2 - simulation, 3 - hardware,
                                 //                  1 - simulation w/o MTBAPI
  unsigned short     dev_type;
                // currently 6 types are supported : 0 - Axio Observer.Z1, 1 - Axio Examiner,
                //           2 - Axio Imager.Z1, 3 - Axio Imager.Z2, 4 - Axio Scope mot. LSM,
                //           5 - Axio Observer.Z1 / 7
  double             start_pos;
  double             step_width;
  double             min_pos;
  double             max_pos;
  double             curr_pos;
  BHPanelAttr        pnl_attr;
  unsigned long      no_of_Zplanes;
  long               lreserve2;
  double             dreserve;
  }AxioParam;


typedef struct AxioLens{        // structure for Axio Lens info
  char               name[264];
  double             magnification;
  double             aperture;
  int                immersionType;
  int                contrastMethod;
  double             workingDistance;
  double             TIRFAngleCorrection;
  int                features;

  }AxioLens;


//  USB Sync Delay device

typedef struct{        // structure for B&H USB Sync Delay device for all modules
  short              connected;  // SyncDel in use ( 4 devices * 4 bits for all modules )
  unsigned char      dev_version[MAX_SPC];   // device versions
  unsigned short     ser_no[MAX_SPC];   // device serial numbers
  float              delay[4];          // devices delay
  BHPanelAttr        pnl_attr;
  short              was_connected;
  short              active_input;      // bits 0-3 - 4 bits for all modules
  double             dreserve;
  }SyncDelParam;


// DDG-210 module parameters

typedef struct{        // structure for PLIM measurement support using DDG-210 module
  short              connected;     // DDG in use - bitwise module number of connected DDG
  unsigned char      ser_no[16];    // serial number of connected DDG
  short              outputs;       // output numbers of control pulses - 6-bit value
                                    //    bits 0..2 - laser pulse, bits 3..5  routing pulse
                                    //    output number in range 1..6
  short              polarity;      // polarity of control pulses - 2-bit value
                                    //    bit 0 - laser pulse, bit 1 - routing pulse
                                    //    bit = 0 negative , = 1 positive,
  short              trigger;       /*  external trigger condition
                                        bits 1 & 0 mean :
                                          00 - ( value 0 ) external trigger disabled (default),
                                          01 - ( value 1 ) external trigger active low,
                                          10 - ( value 2 ) external trigger active high
                                        bit 2 = 1 ( value 4 ) - trigger each cycle
                                              = 0   - trigger only sequence       */
  float              trig_threshold;   // trigger threshold level
                                       //   (-1.0 ... 1.0 V , default 0.1 V)
  float              ldel_time;     // Laser delay time in µs
  float              lwidth_time;   // Laser width time in µs
  float              rdel_time;     // Routing delay time in µs
  float              rwidth_time;   // Routing width time in µs
  BHPanelAttr        pnl_attr;
                                      //   - now all rest channels have the same settings
                                      //   but in future - can be different
  float              del_time[6];     // delay time in µs for all 6 channels
  float              width_time[6];   // width time in µs for all 6 channels
  float              repeat_time;     // repeat sequence after the time, used when trigger is disabled
  float              freserve1;
  double             dreserve1;
  }DdgParam;



//    USB Laser Box MLA4

typedef struct{        // structure for Laser inside MLA4
  short   source;      //  0 - INT, 1 - EXT1, 2 - EXT2
  short   on;          //  bit 0 - laser off/on (internal),
              //  bit 3 - source of Laser On signal : 0 - internal, 1 - external
  short   atten;       //  0 - attenuator off, 1 - attenuator on
  short   cascade;     //  0 - Bypass, 1 - Cascade
  float   power;       //  0 - 100 %
  short   sync_spc;    //  SYNC to SPC 0..3 , bit 0 - SPC1, bit 1 - SPC2
  int     ireserve;
  }MlaLaserPar;



typedef struct{        // structure for B&H USB Laser Box MLA4
  MlaLaserPar        laser_par[4];    // MLA Laser settings
  short              rep_rate;        // Repetition rate for all INT lasers
                                      // 0 - 20, 1 - 50, 2 - 80 MHz, 3 - Continuous
  }BHUSB_MlaData;




typedef struct{        // structure for B&H USB Laser Box MLA4
  short              connected;       // MLA in use - module number of connected MLA ( 1 .. 4)
  unsigned char      dev_version;     // device version
  unsigned short     ser_no;          // device serial number
  BHUSB_MlaData      mladata;         // MLA hardware parameters
  BHPanelAttr        pnl_attr;
  short              was_connected;
  int                ireserve;
  double             dreserve;
  }MlaParam;


//  DCC-100 module parameters

typedef struct Dccdata{          /* structure for DCC module data  */

  unsigned short base_adr;  /* base I/O address on PCI bus */
  short active;            /* most of the library functions are executed
                              only when module is active ( not 0 )*/
  short c1_p5V;            /* Connector 1 +5V  On/Off  */
  short c1_m5V;            /* Connector 1 -5V  On/Off  */
  float c1_gain_HV;        /* Connector 1 Gain/HV [%]  */
  short c1_p12V;           /* Connector 1 +12V On/Off  */
  short c2_p5V;            /* Connector 2 +5V  On/Off  */
  short c2_m5V;            /* Connector 2 -5V  On/Off  */
  short c2_p12V;           /* Connector 2 +12V On/Off  */
  short c2_digout;         /* Connector 2 Digital Outputs State, 0 - ff(hex)  */
  short c3_p5V;            /* Connector 3 +5V  On/Off  */
  short c3_m5V;            /* Connector 3 -5V  On/Off  */
  short c3_p12V;           /* Connector 3 +12V On/Off  */
  float c3_coolVolt;       /* Connector 3 Cooler Voltage ( 0 - 5V)  */
  float c3_coolCurr;       /* Connector 3 Cooler Current Limit ( 0 - 2A)  */
  float c3_gain_HV;        /* Connector 3 Gain/HV [%]  */
  short c3_cooling;        /* Connector 3 Cooler On/Off  */
  }DCCdata;




typedef struct{        // structure for connected DCC-100 modules ( 1 .. 4)
  short              connected;     // bits 0..3 - every bit represents one connected DCC-100
                                    //     current state
  DCCdata            dccdata[4];    // parameters  for 4 DCC-100 modules
  BHPanelAttr        pnl_attr;
  float              gain_C1_limit[4]; //  0..100 %, C1_GAIN_HV for 4 DCC-100 modules
  float              gain_C3_limit[4]; //  0..100 %, C3_GAIN_HV for 4 DCC-100 modules
  short              enable_outputs_on_startup;  // bits 0..3 for 4 DCC-100 modules
  int                lock_setup;     //      bits 0  - Connector1, 1 - C2, 2 - C3, 3 - not defined
                                     //    4 bits per one of up to 8 modules
  short              was_connected; // bits 0..3 - every bit represents one connected DCC-100
                                    //    state from last loaded .set/.sdt file
  char               dig_out_type[4];  // digital outputs type for 4 DCC-100 modules
                                       //  bit = 0 - switch, bit = 1 - key
  char               dig_out_reverse[4];  // reverse digital outputs for 4 DCC-100 modules
                      //  bit = 0 - no change, bit = 1 - button state 0 means 1 in hardware
  char               creserve[4];
  }DccParam;



typedef struct {          /* structure for DCU-USB module data  */

  short c1_p5V;            /* Connector 1 +5V  On/Off  */
  short c1_m5V;            /* Connector 1 -5V  On/Off  */
  short c1_p12V;           /* Connector 1 +12V On/Off  */
  short c1_cooling;        /* Connector 1 Cooler On/Off  */
  float c1_gain_HV;        /* Connector 1 Gain/HV [%]  */
  float c1_coolVolt;       /* Connector 1 Cooler Voltage ( 0 - 5V)  */
  float c1_coolCurr;       /* Connector 1 Cooler Current Limit ( 0 - 2A)  */
  short c2_p5V;            /* Connector 2 +5V  On/Off  */
  short c2_m5V;            /* Connector 2 -5V  On/Off  */
  short c2_p12V;           /* Connector 2 +12V On/Off  */
  short c2_cooling;        /* Connector 2 Cooler On/Off  */
  float c2_gain_HV;        /* Connector 2 Gain/HV [%]  */
  float c2_coolVolt;       /* Connector 2 Cooler Voltage ( 0 - 5V)  */
  float c2_coolCurr;       /* Connector 2 Cooler Current Limit ( 0 - 2A)  */
  short c3_p5V;            /* Connector 3 +5V  On/Off  */
  short c3_m5V;            /* Connector 3 -5V  On/Off  */
  short c3_p12V;           /* Connector 3 +12V On/Off  */
  short c3_cooling;        /* Connector 3 Cooler On/Off  */
  float c3_gain_HV;        /* Connector 3 Gain/HV [%]  */
  float c3_coolVolt;       /* Connector 3 Cooler Voltage ( 0 - 5V)  */
  float c3_coolCurr;       /* Connector 3 Cooler Current Limit ( 0 - 2A)  */
  short c4_p5V;            /* Connector 4 +5V  On/Off  */
  short c4_m5V;            /* Connector 4 -5V  On/Off  */
  short c4_p12V;           /* Connector 4 +12V On/Off  */
  short c4_cooling;        /* Connector 4 Cooler On/Off  */
  float c4_gain_HV;        /* Connector 4 Gain/HV [%]  */
  float c4_coolVolt;       /* Connector 4 Cooler Voltage ( 0 - 5V)  */
  float c4_coolCurr;       /* Connector 4 Cooler Current Limit ( 0 - 2A)  */
  short c5_digout;         /* Connector 5 Digital Outputs State, 0 - ff(hex)  */
  short active;            /* most of the library functions are executed
                              only when module is active ( not 0 )*/
  }DCCUSBdata;



#define NO_DCCUSBPAR_NAMES   20     // no of controls ( per module) with changeable names

typedef struct{        // structure for connected DCC-USB modules ( 1 .. 4)
  short              connected;     // bits 0..3 - every bit represents one connected DCC-USB
                                    //     current state
  DCCUSBdata         dcudata[4];    // parameters  for 4 DCC-USB modules   336B
  BHPanelAttr        pnl_attr;      //  20B
  short              gain_C1U_limit[4]; //  0..100 %, C1_GAIN_HV for 4 DCC-USB modules
  short              gain_C2U_limit[4]; //  0..100 %, C1_GAIN_HV for 4 DCC-USB modules
  short              gain_C3U_limit[4]; //  0..100 %, C1_GAIN_HV for 4 DCC-USB modules
  short              gain_C4U_limit[4]; //  0..100 %, C3_GAIN_HV for 4 DCC-USB modules
  short              enable_out[4];     // bits 0..4 for 4 DCC-USB modules
  int                lock_setup[4];     //      bits 0  - Connector1, 1 - C2, 2 - C3, 3 - C4, 4 - C5
                                     //    5 bits per one of up to 4 modules
  short              was_connected; // bits 0..3 - every bit represents one connected DCC-USB
                                    //    state from last loaded .set/.sdt file
  char               dig_out_type[4];  // digital outputs type for 4 DCC-USB modules
                                       //  bit = 0 - switch, bit = 1 - key
  char               ctrl_name[4][NO_DCCUSBPAR_NAMES][12];      // 960B
  char               dig_out_reverse[4];  // reverse digital outputs for 4 DCC-USB modules
                      //  bit = 0 - no change, bit = 1 - button state 0 means 1 in hardware
  char               reserve[664];     //   sum = 1384B, keep total size of 2048B
  }DcuParam;




typedef struct{        // structure for Ti:Sa Laser & AOM device
  unsigned char      in_use;           // control in use or not  , b0 - AOM, b1 - known laser,
                                       //    b2 - SHM
  unsigned char      was_in_use;
  unsigned char      AOM_status;       //  AOM status :bit 0 - used or not, b1 - present,
                                       //              b2 - modulation, b3 - remote, b4 - error,
  unsigned char      AOM_power;        //  AOM power in %  0 - 100
  unsigned long      AOM_freq;         //  AOM frequency in Hz
  char               AOM_ser_no[8];    // AOM serial number
  unsigned char      laser_type;       //  b0-3 - laser type 0-15, 0 - unknown
  unsigned short     laser_status;     //  laser status :bit 0 - used or not, b1 - present,
                   //  b2 - on/off, b3 - shutter(tuneable), b4 - warm up, b5 - pulsing,
                   //  b6 - error, b7-10 - laser type 0-15, 0 = unknown
      //  Insight X3 specific  b11 - ready to turn on, b12 - turning on, b13 - turned on
      //                       b14 - FixedIR shutter
  unsigned short     laser_wav;        //  current laser wavelength in nm
  unsigned short     wav_lo_limit;     //  laser wavelength low limit in nm
  unsigned short     wav_hi_limit;     //  laser wavelength high limit in nm
  char               laser_id[64];      // laser identification string
  char               laser_ser_no[64];  // laser serial number
  BHPanelAttr        pnl_attr;
  float              first_diff_angle; // value of 1st range diffraction angle [mrad]
                                       //   value is fixed for the current AOM device
                                       //   used to calculate AOM frequency
  unsigned char      AOM_type;         //  b0 = 0  AOM1/IR, 1 - AOM2/IR+
                                       //  b7 = 1  AOM version >= 2.1.0 - it means that 'space' character is required between CMD and parameter
                                       //  b7 = 0  AOM version < 2.1.0 -  no 'space' character is required between CMD and parameter
  unsigned char      AOM_mode;         //  bit 0 = 0 - AOM only,  = 1 - AOM+SHG
                                       //  when b0 = 1 : bit 1 = 0 -  MP Mode, = 1 - 1p confocal AOM+SHG
  unsigned char      SHG_status[4];    //  SHG status :  byte 0 - bit 0 - used or not, b1 - present,
                                       //                         bit 2 - Shut1, bit3 - Shut2
                                       //    byte 1-3 filled by read status ( GST)
  unsigned char      GDD_mode;         //  for InSight X3 laser, b0 = 0  manual, 1 - automatic
  float              GDD_val;          //                     actual motor position 0.00 - 100.00
  char               SHG_ser_no[8];    //  SHG serial number
  unsigned short     AOM_wav;          //  current AOM wavelength in nm - can be different than laser_wav
  unsigned short     SHG_wav;          //  current SHG wavelength in nm
                                     //
  char               reserve[5];
  }AomParam;



#define MAX_NO_OF_LENS   8

typedef struct{        // structure for Prior motor stage calibration
  double   range_x;                         // max X size in [mm]
  double   range_y;                         // max Y size in [mm]
  unsigned short    speed;                  // speed of movement   1 - 100  during measurement
  double   step_wx[MAX_NO_OF_LENS];         // X step width for Zoom = 1  in [µm] 1 .. 20000
  double   step_wy[MAX_NO_OF_LENS];         // Y step width for Zoom = 1  in [µm] 1 .. 20000
  char     lname[MAX_NO_OF_LENS][64];       // user defined names of Objective Lens. 8*64 = 512B
  short    lens_in_use;                     // bits 0-7 tell which lens are used, not used - empty name
  short    dflt_in_use;                     // data from  dflt_cal_par
  char     remarks[256];                    // remarks
  double   skew_angle;                      //  skew angle[°] -10 .. 10, default 0
  char     reverse_axes;             // bits 0 -X, 1 -Y, 2 -Z axis reversed (1) or not (0) for stage
  double   xy_resolution;                   // resolution of 1step for stage axes (X & Y) [µm]
  double   z_resolution;                    // resolution of 1step for Z axe [µm]
  char     focus_present;                   // focus ( Z axis ) 0 - not present, 1 - FB20X connected,
                                            //                  2 - NORMAL focus connected
  char     joystick_connected;
  char     reverse_joystick_axes;    // bits 0 -X, 1 -Y, 2 -Z axis reversed (1) or not (0) for joystick
  double   max_z;                           // max Z size in [µm]  ( StageParam.min_z_pos .. StageParam.max_z_pos )
  char     sync_corner;              // corner to which the stage moves during initial synchronisation ( SIS or RIS)
                                     //    1= Left-Forward, 2= Right-Forward, 3= Right-Back, 4= Left-Back
  double   min_z;                    // minimum Z position in [µm]   ( StageParam.min_z_pos .. StageParam.max_z_pos )
  unsigned char zdrive_type;         // 0 - not present : z_resolution, min_z, max_z controlled not used
                                     // > 0 min_z_pos, max_z_pos from device specification
                                     //  2 - Nikon TE2000-U
  unsigned short    zspeed;                 // speed of Z movement   1 - 100%  during measurement
  char     stage_present;                   // stage ( XY axis ) 0 - not present, 1 - present
  double   min_xy_resol;                    // min xy resolution - for SS = 1  [µm]
  double   min_z_resol;                     // min z resolution - for SSZ = 1 [µm]
  char     reserve[41];                     // keep total size of 1024B
  }StageCalParam;


typedef struct{        // structure for Prior motor stage ( OptiScan III ) device
  unsigned char      in_use;           //
  unsigned char      connected;        // bit 1 - connected, bit 0 - present
  unsigned char      was_connected;    //  when loading params from a file
  unsigned char      at_limit;         //  is at limit: bits 0,1 - X+,X-, bits 2,3 - Y+,Y-, bits 4,5 - Z+,Z-
  unsigned char      active_lens;      //  number of used lens 0-5
  unsigned int       ser_no;           //  serial number
  char               prior_info[512];  //  buffer for motor stage info
  BHPanelAttr        pnl_attr;         //  size = 20B
  StageCalParam      cal_par;          //  motor stage calibration parameters, size = 1024B
  unsigned char      step_type;        //  0 - manual defined, 1 - from scanner ( current lens step width average )
  double             manual_x_step;    //  length (in µm) of manual defined step in X dir., 1 .. 20000 for stage
  unsigned short     speed;            //  speed of movement   1 - 100  for manual stage moves
  short              size_x;           //  no of elements in X dir. 1 .. MAX_MOSAIC_SIZE
  short              size_y;           //  no of elements in Y dir. 1 .. MAX_MOSAIC_SIZE
  short              frames_per_el;    //  frames per element 1 .. 32767
  long               start_xpos;       //  start x position in mikroticks
  long               start_ypos;       //  start x position in mikroticks
  double             max_x;            //  max move in X dir. [mm]
  double             max_y;            //  max move in Y dir. [mm]
  int                mikrosteps_per_mikron;
  long               start_zpos;       //  start z position in mikroticks
  unsigned short     zspeed;           //  speed of Z movement   1 - 100  for manual Z moves
  double             manual_z_step;    //  length (in µm) of manual defined step, 0.001 .. 20000 for Z movements
  double             max_z_pos;        //  maximum Z position [µm]   - device limit
  double             manual_y_step;    //  length (in µm) of manual defined step in Y dir., 1 .. 20000 for stage
  char               rev_tile_x;       //  reverse direction of tiling X
  char               rev_tile_y;       //  reverse direction of tiling Y

  short              zstack_enabled;   //  Z stack enabled 0,1
  short              zstack_type;      //  Z stack type :  0 - Z Planes ( w/o mosaic),
                                       //                  1 - Z planes with Tile Mosaic, 2 - Z Planes Mosaic
  short              zsize_x;          //  no of elements in X dir. 1 .. MAX_MOSAIC_SIZE for Z Planes mosaic
  short              zsize_y;          //  no of elements in Y dir. 1 .. MAX_MOSAIC_SIZE for Z Planes mosaic
  short              zframes_per_el;   //  frames per element 1 .. 32767   for Z Planes mosaic
  short              zautosave;        //  0, 1 each cycle, 2 last cycle only ( end of meas.)
  double             zstart;           // start position (in µm) of Zstack
  double             zend;             // end   position (in µm) of Zstack
  double             zstep;            // length (in µm) of Zstack step
  short              zplanes_no;       //  Z planes number  1 .. 32767   for Zstack
  short              fix_zpar;         // fix Zstack parameter: bit 0 - End pos., 1 - Step length, 2- no of Zplanes
  double             min_z_pos;        //  minimum Z position [µm]  - device limit
  short              tile_imaging_enabled;   //  tile imaging enabled 0,1
  char               reserve[108];     //  keep total size of 1792B
  }StageParam;




#define NO_SIS_NAMES   20     // no of changeable names inp/out names ( per module)

typedef struct {          /* structure for SIS-2x4/4x4 module data  */

  short active;         /* most of the library functions are executed only when module is active (not 0),
                               can be set only during initialisation */
  short switch_a;       /* state of switch A - bits 0..3 = state of outputs 1..4, default 0 */
  short switch_b;       /* state of switch B - bits 0..3 = state of outputs 1..4, default 0  */
  short switch_c;       /* state of switch C (SIS4x4 only) - bits 0..3 = state of outputs 1..4, default 0  */
  short switch_d;       /* state of switch D (SIS4x4 only)  - bits 0..3 = state of outputs 1..4, default 0  */
  short mult_con;       /* multiple input/output connections allowed (1), or not (0), default 0
                            bit 0 - switch A, bit 1 - switch B, bit 2 - switch C, bit 3 - switch D  */
  short equal_mod;      /* equal switches settings for all active modules (1), or not (0), default 0
                               parameter has the same value for all active modules */
  short xor_a_b;        /* XOR setting of outputs A & B (1), or not (0), default 0*/
  short xor_c_d;        /* XOR setting of outputs C & D (1), or not (0), default 0,  (SIS4x4 only) */
  char reserve[18];

  }SISdata;


typedef struct{        // structure for connected SIS modules ( 1 .. 4)
  short              connected;     // bits 0..3 - every bit represents one connected SIS
                                    //     current state
  SISdata            sisdata[4];    // parameters  for 4 SIS modules   4*18 = 72B
  BHPanelAttr        pnl_attr;      //  20B
  short              enable_out[4];     // bits 0..1 for 4 SIS modules
  short              was_connected; // bits 0..3 - every bit represents one connected SIS
                                    //    state from last loaded .set/.sdt file
  short              options;       //    bit 0 - Scale GUI contents on resize
                                    //    bit 1 - GUI always on top
                                    //    bit 2 - do not change inp/out names when loading setups

  char               ctrl_name[4][NO_SIS_NAMES][16];      // 4 * 20 * 16 = 1280B
  char               reserve[590];     //   keep total size of 2048B
  }SisParam;



typedef struct {          /* structure for SIS-2x4 module data  */

  short active;         /* most of the library functions are executed only when module is active (not 0),
                               can be set only during initialisation */
  short switch_a;       /* state of switch A - bits 0..3 = state of outputs 1..4, default 0 */
  short switch_b;       /* state of switch B - bits 0..3 = state of outputs 1..4, default 0  */
  short mult_con;       /* multiple input/output connections allowed (1), or not (0), default 0
                            bit 0 - switch A, bit 1 - switch B  */
  short equal_mod;      /* equal switches settings for all active modules (1), or not (0), default 0
                               parameter has the same value for all active modules */
  short xor_a_b;        /* XOR setting of outputs A & B (1), or not (0), default 0*/
  }OldSISdata;   // before adding SIS4x4



typedef struct{        // structure for connected SIS modules ( 1 .. 4)
  short              connected;     // bits 0..3 - every bit represents one connected SIS
                                    //     current state
  OldSISdata          sisdata[4];    // parameters  for 4 SIS modules   4*8 = 36B
  BHPanelAttr        pnl_attr;      //  20B
  short              enable_out[4];     // bits 0..1 for 4 SIS modules
  short              was_connected; // bits 0..3 - every bit represents one connected SIS
                                    //    state from last loaded .set/.sdt file
  short              options;       //    bit 0 - Scale GUI contents on resize
                                    //    bit 1 - GUI always on top
                                    //    bit 2 - do not change inp/out names when loading setups

  char               ctrl_name[4][10][16];      // 4 * 10 * 16 = 640 B
  char               reserve[314];     //   sum = 710B, keep total size of 1024B
  }OldSisParam;         //  size changed 9.10.23





typedef struct{        // structure for extension of external devices setup  parameters
  int            edev_size[8];   // up to 8 structures with different sizes
                        //   can follow for different external devices
                        //   1st implemented is DcuParam for DCU - USB Detector Control
  DcuParam       dcu_par;       //  parameters for DCU - USB Detector Control
  SisParam       sis_par;       //  parameters for SIS2x4/4x4 - USB Signal Switch
  }ExtDevSetup2;




typedef struct{        // structure for external devices setup  parameters
  int            edev_size[8];   // up to 8 structures with different sizes
                        //   can follow for different external devices  (v.8.91 )
                        //   1st implemented is AxioParam for Zeiss Axio Observer.Z1
  AxioParam      axio_data;     //  parameters for Z scan using Zeiss Axio Observer.Z1
  SyncDelParam   sdel_data;     //  parameters for B&H USB SyncDel device
  DdgParam       ddg_par;       //  parameters for PLIM support using DDG-210 module
  MlaParam       mla_par;       //  parameters for USB Laser Box MLA4
  DccParam       dcc_par;       //  parameters for DCC-100 module
  AomParam       aom_par;       //  parameters for Ti:Sa Laser & AOM control
  StageParam     prior_par;     //  parameters for Prior motor stage ( OptiScan III )
  ExtDevSetup2   dev_setup2;    //  extension of ExtDevSetup for future devices
  }ExtDevSetup;




typedef struct{        // structure for MCS Image parameters
  int             active;             // MCS Image is calculated
  Window          mtwin;              // time windows for MCS images
  unsigned short  mcsta_points;       // no of MCS_TA values
  unsigned short  mcsta_flags;        // MCS_TA flags   bit 0 = 1 - use 'invalid' photons,
                                      //      bit 1-2  =  marker no used as trigger
                                      //      bit 3-5  =  1..7   4 ** value = points number
  unsigned int    mcsta_tpp;          // MCS_TA Time per point  in Macro Time units
                                      // time per point[s] = mcsta_tpp * mt_resol( from MeasFCSInfo)
  float           offset;             // for MCS_TA: offset in percent  -100 .. +100
  float           tpp;                // time per point in [ms]
  }MIMGParam;




// Moments traces parameters ( M0, M1, M2 )
typedef struct  {
  short    active;         // (0)1 - (not) active
  short    module;
  short    curve;          // routing channel no  ( 1 - 16 )
  short    T_wind;         // number of T window interval used in intensity ( M0) mode
  int      color;          // trace color converted  to Windows style
  int      points;         // number of points in trace
  }MOMtrace;  //



typedef struct{        // structure for M0, M1, M2  histogram windows
  int                active;   // window visible
  unsigned short     mom_type; // additional parameters -  not used yet
  unsigned short     par_ext;  // parameter extension
  float              time;     // in [ms], time per point
  BHPanelAttr        pnl_attr;
  Disp2DAttr         d2para;
  Window             mtwin;       // time gate windows for M0 traces
                     // extra parameters
  int                no_of_steps;
  int                ipar_ext2;
  float              fpar_ext;
  MOMtrace           htrace[MAX_NO_OF_TRACES];
  }MOMParam;



typedef struct{        // structure for extension of system parameters
  short  saved_in64b;   // parameters ( and file ) were saved in 64-bit SPCM version
  int    adc_scan;      // ADC resolution for scan modes
  int    adc_fimg;      // ADC resolution for FifoImage modes
  int    adc_norm;      // ADC resolution for all other modes
  short  img_ncx, img_ncy, img_x, img_y;  // Image size params for FifoImage mode
  short  fifo_ncx;
  short  norm_ncx, norm_ncy;
  short  wf_ncx, wf_x, wf_y;  // Image size params for Wide-Field Image mode
  int    adc_wf;
  short  flags;         // bit 0 - Repeat Time = Coll. Time, bit 1 - Display Time = Coll. Time,
  short  skip_2nd_line_clk; // Fifo Image ( some microscopes delevers Line clock
                           //             at the beg. and at the end of the line )
  short  scan_type;        // 0 - Unidir, 1 - Bidir
  unsigned int    right_border;     // for Bidir scanning
  unsigned int    left_border;      // for FifoImage mode instead of SP_BORD_L
  float  tdc_range_norm, tdc_range_fifo, tdc_range_fimg; // TDC Range ( SPC-QC-x0x module) in normal(not fifo), fifo & fifo image modes
  float  tdc_range_fifo_abs;    // TDC Range ( SPC-QC-x0x module) in fifo absolute time mode
  int    adc_fifo_abs;     // ADC resolution for SPC-QC-x0x module in fifo absolute time mode
  char   reserve[188];     // keep fixed size of the structure = 256B
  }SysParExt;


#define MAX_MOSAIC_SIZE   2048

typedef struct{         //  structure for Mosaic parameters
  short  enable;        //  Mosaic Imaging enabled
  short  size_x;        //  no of elements in X dir. 1 .. MAX_MOSAIC_SIZE
  short  size_y;        //  no of elements in Y dir. 1 .. MAX_MOSAIC_SIZE
  short  frames_per_el; //  frames per element 1 .. 32767
  short  type;          //  0 - sequence of frames, 1 - sequence of routing channels,
                        //  2 - sequence of Z planes of Axio Observer.Z1
                        //  3 - sequence of frames with motor stage
                        //  4 - sequence of Z planes with motor stage
  short  restart;       //  0 - no restart, 1- marker 3, 2 - trigger
  short  grid;          //  on/ off
  int    grid_color;
  int    accu_cycles;   //  accumulation cycles when restart > 0
  short  chan_per_el;   //  channels per element 1 .. 256
  short  zplanes_no;    //  Z planes no of Axio Observer.Z1 when type = 2
  char   reserve[38];   // keep fixed size of the structure = 64B
  }MosaicParam;



typedef struct{             //  structure for Wide-field Image mode parameters
  short  Tmod_no;           //  T module number ( 0-2), default 0
  short  Xmod_no;           //  X module number ( 0-2), default 1
  short  Ymod_no;           //  Y module number ( 0-2), default 2
  short  Xmatch_clk[2];     //  no of matching intervals for X module , 0 .. 10,values before and after
  short  Ymatch_clk[2];     //  no of matching intervals for Y module , 0 .. 10,values before and after
  short  tac_gain_Tmod;     //  TAC gain for T module - not used, instead TAC_G parameter is used
  short  tac_gain_XYmod;    //  TAC gain for XY modules,   1 .. 15
  float  tac_range_Tmod;    //  TAC range for T module - not used, instead TAC_R parameter is used
  float  tac_range_XYmod;   //  TAC range for XY modules,  50 .. 5000 * 1e-9 [s]
  short  size_x;            //  no of pixels in X dir. 16 .. 4096
  short  size_y;            //  no of lines in Y dir. 16 .. 4096
  short  adc_re;            //  no of points in decay curve   1 .. 4096
  short  ncx;               //  no of routing channels   1 .. 16
  char   reserve[30];       // keep fixed size of the structure = 64B
  }WFParam;


typedef struct{             //  structure for layout of windows on the desktop
  int    desktop_width;
  int    desktop_heigth;
  int    no_of_monitors;
  int    first_mon_width;
  int    first_mon_height;
  BHPanelData  pnldata[35]; //  10 more then MAX_PNL
  char   reserve[48];       // keep fixed size of the structure = 1600B
  }WndLayout;


#define MAX_NO_OF_3DGR  8                // max number of 3D graph windows
#define MAX_NO_OF_GR  MAX_NO_OF_3DGR     // max number of graph windows = 3D graph windows (Decay3D is separate window )

#define TRPAR_VERSION 1     // current TrParExt version

typedef struct{         // structure for extension of 3D trace parameters
  char   version;         // = 0 - not valid, > 0 - valid & means version number
  char   tr3d_name[MAX_NO_OF_GR][32];   // user defined names of 3D traces, 8*32 = 256B
  union {
    unsigned char scale_bar;        // image scale bar : bit 0 = 0 - off, 1 - on ( visible)
    struct{                         // bits 5-4 system type = 00 - not defined,
      unsigned char sbEnable : 1;   //  01 - GVD on motorized Axio,
      unsigned char : 3;            //  10 - GVD on manual Axio - currently not available
      unsigned char sbSysType : 2;  //  11 - LSM980, no GVD
    };
  };
  char   image_size_source;     // 1 - user defined or from loaded file,
                                // 0 - calculated from scanner & objective params
  char   scope_connected;  // microscope mode = 0 - not connected, 1 - simulation w/o MTBAPI,
                           //                   2 - simulation,    3 - hardware,
  float  lens_magnifier;
  float  zoom_factor;      // scanner zoom, = 0 - gvd not in use
  float  FOV_at_zoom_1;    // scanner FOV at zoom = 1 , in µm
  float  image_size;       // used image size, in µm
                           //  0 - not valid
                           //  = FOV_at_zoom_1 / zoom_factor / lens_magnifier
                           //    if values available and image_size_source = 1
  char   scope_name[32];   // name of connected microscope
  char   lens_name[64];    // name of lens used with connected microscope
  unsigned char   no_of_user_lens;  // 0 ..6  number of user defined objectives
  unsigned char   curr_user_lens;   // 0 ..6  current number of used user objectives
  float  user_lens_magnifier[6];
  char   user_lens[6][16]; // list of user objectives names
                           //  name should contain magnifier at the beginning ( e.g.  2.5x)
  char   reserve[18];     // keep fixed size of the structure = 512B
  }TrParExt;


typedef struct{         // structure for correction data for multichannel detector
  char   enable;            //  detector correction is enabled
  float  corval[16];        //  correction values for 16 channels
  char   cor_fpath[260];    // correction file path
  char   det_type[64];      // detector type
  char   det_sno[64];       // detector serial number
  char   remarks[192];      // remarks
  char   reserve[123];      // keep fixed size of the structure = 768B
  }CorPar;




typedef struct{         // structure for parameters of 3D Lifetime traces
  char   tr_no;         //  no of 3D trace which uses this structure ( 0-7)
  char   reverse_col;   //  reverse color bar of lifetime
  double min_lifetime;  //  minimum of lifetime range ( for blue), from 0
  double max_lifetime;  //  maximum of lifetime range ( for red ), up to 10000ps
  double m1_irf;        //  1st moment of IRF ( Instrument Response Function)
                        //  either calculated from ref_lifetime and ref_fpath file,
                        //            or typed in manually
  double ref_lifetime;  //  Intensity weighted lifetime of ref_fpath data file
                        //    either calculated from ref_fpath file,
                        //            or typed in manually
  char   ref_fpath[260];    // file path of reference data file ( to calculate Ref. Lifetime)
  unsigned long min_ph_no;            //
  unsigned long max_ph_no;            //  when binning used - binned value
  char   no_intens_overlay;    // = 1 do not use intensity overlay
  char   brightness;           //  0 .. 100 %
  char   contrast;             //  0 .. 100 %
  char   binning;              // use binning or not
  unsigned long min_ph_no_bin;        //
  unsigned long max_ph_no_bin;        //  when binning used - not binned value
  unsigned long min_ph_no_int;        //  for LIFE_INT trace - minimum intensity ( photons number)
  unsigned long max_ph_no_int;        //  for LIFE_INT trace - maximum intensity ( photons number)
  struct {
    char bi_color_code : 1;
    char : 7;
  };
  char   reserve[189];      // keep fixed size of the structure = 512B
  }LifeTrPar;



// traces parameters for 3D Histograms ( TYPE_DECAY3D )
typedef struct  {
  short    exist;          // bit 0 - exist(defined), bit 1 - visible on trace23_panel
  short    show;           // bit 0 - (0)1 - (not)show, bit 1 - disabled ( d3_mode of parent3Dwnd != 2 (F(x,y))
  int      color;          // trace color converted  to Windows style
  short    parent3Dwnd;    // no of parent 3D window
  short    type;           //  0 - data for 1 point on parent window, 1 - data from region on parent wnd,
                           //  2 - all visible pixels on the parent wnd
  short    binning;        // binning parameter for type = 0
                           //      0 - binning not used, n > 0 - bin ( 2n+1)*(2n+1) points , max 10
  Interval x_coord;        // region X coordinates from-to ( X1 - X2 )
  Interval y_coord;        // region Y coordinates from-to ( Y1 - Y2 )
  unsigned int    pt_x;           // point x coordinate for type = 0
  unsigned int    pt_y;           // point y coordinate for type = 0
  short    cur_bind;       // type of binding with cursors, 0 - no, 1 - cursors -> TrParam,
                           //      2 - TrParam -> cursors, 3 - both directions
  short    init_bind;      // initial type of binding with cursors
                           //   ( after loading from file, or before switching off cursors)
  short    prepare_data;   //  preparing data required ( after coordinates change during meas.)
  char     reserve[20];    // for future use, keep fixed size of the structure = 64B
  }HST3Dtrace;


typedef struct{        // structure for FCS, FIDA, FILDA MT histogram windows
  int            active;   // window visible, histogram is calculated
  unsigned short        hst_type; // additional histogram parameters
                           // for Decay3d : 0 - photon, 1 - normalize to maximum,
                           //               2 - normalize to integral
  unsigned short        par_ext;  // parameter extension
  float          time;     // in [ms], not used yet
  BHPanelAttr    pnl_attr;
  Disp2DAttr     d2para;
  HST3Dtrace     h3dtrace[MAX_NO_OF_TRACES];
  short          no_of_traces;  // number of existing traces
  short          no_of_vis_traces;  // number of visible traces (on trace23_panel)
  short          undo_possible;     // undo last delete action is possible
                 // extra parameters for future use
  DparVal maxcnt_m;   // for hist_type 0 - photon
  DparVal lbline_m;
  DparVal bline_m;
  DparVal maxcnt_d;   // for hist_type 1,2 - normalize
  DparVal lbline_d;
  DparVal bline_d;
  char reserve[74];        // for future use, keep fixed size of the structure = 1280 B
  }HST3DParam;



#define MAX_DCC     4            // max number of DCC modules controlled by SPCM
#define DCCPAR_VERSION 1         // current DccParExt version
#define NO_DCCPAR_NAMES   16     // no of controls with changeable names

typedef struct{         // structure for extension of DCCParam structure
  char   version;         // = 0 - not valid, > 0 - valid & means version number
  char   dcc_ctrls_name[MAX_DCC][NO_DCCPAR_NAMES][12];   // user defined names of DCC controls
                                // for 4 DCC modules   4 * 16 * 12 = 768B
  char   reserve[255];     // keep fixed size of the structure = 1024B
  }DccParExt;


#define MAX_NO_OF_COLORS   11 // max value for DI_COL_NO

typedef struct
{
  int   offsx;             //  DI_3DOFFX - offset X for 3D display
  int   offsy;             //  DI_3DOFFY - offset Y for 3D display
  int   inclination;       //  DI_3DINCX - inclination X for 3D display
  int   body_color;        //  DI_BOD_C
  int   curve_color;       //  DI_3DCOL
  int   d3_mode;           //  DI_3DMODE
  int   ywin;              //  DI_YWIN
  int   xwin;              //  DI_XWIN
  int   twin;              //  DI_TWIN
  int   sywin;             //  DI_SYWIN
  int   sxwin;             //  DI_SXWIN
  int   d3_style;          //  DI_3DSTYL
  int   interpolate;       //  DI_INTCOL- bit 0, DI_INTPIX - bit 1
  int   col_no;            //  DI_COL_NO
  int   color[MAX_NO_OF_COLORS];   // 0 - DI_COLMIN, 1 - DI_COLMAX, 2..5 - DI_COL2-5,
                           // 6 - 10 are stored in BIN_SETUP
  int   col_hi;            //  DI_HICOL
  int   detail;            //  DI_DETAIL
  int   ascale;            //  DI_ASCALE
  int   lcol_no;           //  DI_COL_NO  for log scale
} Disp3DAttr;



typedef struct _BHInterval {
    unsigned int    from;
    unsigned int    to;
    }BHInterval;



typedef struct _Trace3d {
  short    log_scale;       /*  display type 0- linear , 1- logarithmic  */
  short    offsx;           /*  display offset X  - parameter DI_3DOFFX  */
  short    offsy;           /*  display offset Y  - parameter DI_3DOFFY  */
  double   inclination;     /*  display inclination- parameter DI_3DINCX  */
  short    mode;            /*  display mode  - parameter DI_3DMODE  */
  unsigned int    page;            /*  page number in SPC memory */
  unsigned int    no_of_curves;    /*  number of curves  */
  unsigned int    no_of_disp_curves;    /*  number of displayed curves */
  unsigned int    no_of_points;    /*  number of points  */
  short    resolution;         /* 0 means 65536 */
  BHInterval * win;            /*  pointer to  window interval - depending on
                                   display mode = DI_XWIN,DI_YWIN,DI_TWIN */
  BHInterval * win_y;          /*  pointer to Y window interval - for f(t,x,y,param) */
  unsigned short  first_disp_point; /*  number of first displayed point on curves  */
  unsigned short  last_disp_point;  /*  number of last displayed point on curves  */
  unsigned int    first_disp_curve; /*  number of first displayed curve  */
  unsigned int    last_disp_curve;  /*  number of last displayed curve  */
  short    reduction;        /*  display reduction ratio  */
  int      color;            /*  colors of curves  (DIS_3DCOL converted ) */
  int      body_color;       /*  color of the body (DIS_BOD_C converted ) */
  double   curve1_xpos;      /*  x position on the graph of the first curve */
  double   curve1_ypos;      /*  y position on the graph of the first curve */
  double   xinc;             /*  x increment */
  double   yinc;             /*  y increment */
  double   ygain;            /*  y gain */
  short    no;               /*  3d trace no */
  char   type;               /*  trace type 0 - Picosecond FLIM or Wide-Field FLIM, */
                             /*               1 - MCS FLIM, 2 - Intensity */
                             //               3 - LIFET (Lifetime),
                             /*              -1 - not active MCS ( in not fifo_image modes )*/
  unsigned char     twin;    /*  time interval no: bits 0-3 for type 0, bits 7-4 for type 1 */
  char   chan_mosaic;        /*  mosaic in use & sequence of routing channels different way of display */
  char   reserve[3];         /* for future use   */
  long     low_offset;       /*  DI_BLINE */
  long     max_count;        /*  DI_MAXCNT */
  BHInterval * swin;           /*  pointer to ScanX(Y) window interval -  depending on
                                 display mode = DI_XWIN,DI_YWIN,DI_TWIN */
  BHInterval * swin_y;         /*  pointer to ScanY window interval  */
  long no_of_blocks;         /*  for get3Dcurve */
  } Trace3d;



typedef struct  {
  short        active;
  unsigned short      tr3d_module; // mod_no for 3D trace, missing in Trace3d structure
  unsigned short      mem_bank;
  short        main_height;   // main panel height
  short        main_width;    // main panel width
  short        main_top;      // main panel top
  short        main_left;     // main panel left
  short        main_ctrl_width;  // width of the panel when ctrls are close to the right border of the panel
  char         main_flags;    //  bit 0 - keep 4:3 aspect ratio for main window
                              //  bit 1 - keep 16:9 aspect ratio
  char         cursors_enabled;
  char         no_of_mod;     //  no of modules used to build image
  char         tdc_chan;      //  Input Channel for SPC-QC-104 module
  char         reserved[4];
  BHPanelAttr  pnl_attr;
  Disp2DAttr   d2para;    // Disp2DAttr for modes with 2D display
  Disp3DAttr   d3para;    // additional parameters for 3D display
  Disp2DAttr   d23para;   // Disp2DAttr for modes with 3D display
  Trace3d      tr3d;      // 3D trace parameters structure
  Disp2DAttr   d21para;   // Disp2DAttr for modes with 2D display and 32-bit range
                          //      ( fifo modes & 2D Block Display )
  }GR_Param;              // Main Graph window parameters


///////////////////////////////////////////////////////////////////////////////////////////

       // for solving problem of compatibility beetwen 32 and 64 bit version

       //   Trace3d structure contains pointers which have different size in 32/64 bit versions
       //       pointers size = 4 bytes when 32bit envir. = 8 bytes when 64bit target
       //   setups created in both versions would be not compatible
       //  to avoid it binary setup contains Trace3d32 in GR_Param32 instead of  Trace3d and GR_Param
       //      !!!!    do not set/load pointers in Trace3d when saving/loading GR_Param

typedef struct {
    short    log_scale;       /*  display type 0- linear , 1- logarithmic  */
    short    offsx;           /*  display offset X  - parameter DI_3DOFFX  */
    short    offsy;           /*  display offset Y  - parameter DI_3DOFFY  */
   double    inclination;     /*  display inclination- parameter DI_3DINCX  */
    short    mode;            /*  display mode  - parameter DI_3DMODE  */
    unsigned int    page;            /*  page number in SPC memory */
    unsigned int    no_of_curves;    /*  number of curves  */
    unsigned int    no_of_disp_curves;    /*  number of displayed curves */
    unsigned int    no_of_points;    /*  number of points  */
    short    resolution;      /* 0 means 65536 */
    unsigned int    win;             /*  originally pointer to  window interval - here empty */
    unsigned int    win_y;           /*  originally pointer to  window interval - here empty */
  unsigned short    first_disp_point; /*  number of first displayed point on curves  */
  unsigned short    last_disp_point;  /*  number of last displayed point on curves  */
    unsigned int    first_disp_curve; /*  number of first displayed curve  */
    unsigned int    last_disp_curve;  /*  number of last displayed curve  */
    short    reduction;        /*  display reduction ratio  */
    int      color;            /*  colors of curves  (DIS_3DCOL converted ) */
    int      body_color;       /*  color of the body (DIS_BOD_C converted ) */
   double    curve1_xpos;      /*  x position on the graph of the first curve */
   double    curve1_ypos;      /*  y position on the graph of the first curve */
   double    xinc;             /*  x increment */
   double    yinc;             /*  y increment */
   double    ygain;            /*  y gain */
    short    no;               /*  3d trace no */
      char   type;             //  trace type 0 - Picosecond FLIM, 1 - MCS FLIM, 2 - Intensity
                               //            -1 - not active MCS ( in not fifo_image modes )
    unsigned char   twin;             //   time interval no: bits 0-3 for type 0, bits 7-4 for type 1
      char   chan_mosaic;      //  mosaic in use & sequence of routing channels different way of display
      char   reserve[3];       //   for future use
     long    low_offset;       /*  DI_BLINE */
     long    max_count;        /*  DI_MAXCNT */
    unsigned int    swin;             /*  originally pointer to  window interval - here empty */
    unsigned int    swin_y;           /*  originally pointer to  window interval - here empty */
    long     no_of_blocks;     /*  for get3Dcurve */
    } Trace3d32;


typedef struct  {
  short        active;
  unsigned short      tr3d_module; // mod_no for 3D trace, missing in Trace3d structure
  unsigned short      mem_bank;
  short        main_height;   // main panel height
  short        main_width;    // main panel width
  short        main_top;      // main panel top
  short        main_left;     // main panel left
  short        main_ctrl_width;  // width of the panel when ctrls are close to the right border of the panel
  char         main_flags;    //  bit 0 - keep 4:3 aspect ratio for main window
                              //  bit 1 - keep 16:9 aspect ratio
  char         cursors_enabled;
  char         no_of_mod;     //  no of modules used to build image
  char         tdc_chan;      //  Input Channel for SPC-QC-104 module
  char         reserved[4];
  BHPanelAttr  pnl_attr;
  Disp2DAttr   d2para;    // Disp2DAttr for modes with 2D display
  Disp3DAttr   d3para;    // additional parameters for 3D display
  Disp2DAttr   d23para;   // Disp2DAttr for modes with 3D display
  Trace3d32    tr3d;      // 3D trace parameters structure
  Disp2DAttr   d21para;   // Disp2DAttr for modes with 2D display and 32-bit range
                          //      ( fifo modes & 2D Block Display )
  }GR_Param32;            // Main Graph window parameters




///////////////////////////////////////////////////////////////////////////////////////////



/*
  4 - Measurement Description Blocks

Each data block can (but need not) have its own system (hardware) parameter set
 which can differ from the setup parameters.
In the block header of each data block a corresponding measurement description block is specified.
Therefore the number of measurement description blocks can vary
 from one (if all stored data blocks originate from only one measurement)
 to the overall number of saved data blocks
  (if all blocks are measured with different hardware parameters).

The number, the length and the location of the measurement description blocks is stored
  in the file header at the beginning of the file.
The information in the measurement description blocks is used for
 the 'Block Info' or 'Set Info' function in the Load, Save and Trace Parameters menus.
If the button 'Use System Parameters from the Selected Block' is pressed,
 the system parameters are replaced by the data in the measurement description block.
The measurement description blocks are stored in a binary format.

From the software version 7.9  measurement description block contains also additional
 information (in the structure MeasStopInfo ) collected at the end of the measurement.
 This information can be useful when analyzing system's wrong setup or malfunction.

From the software version 8.1  measurement description block contains also additional
 information (in the structure MeasFCSInfo ) collected at the end of the FIFO measurement
 when FCS curve were created.
 The information is valid only for data blocks of type FIFO_DATA or FIFO_DATA_FROM_FILE

From the software version 8.3  measurement description block contains also additional
 information (in the structure MeasFIDAInfo ) collected at the end of the FIFO measurement
 when FIDA or FILDA curves were created.
 The information is valid only for data blocks of type FIFO_DATA or FIFO_DATA_FROM_FILE

From the software version 8.4  measurement description block contains instead of MeasFIDAInfo
 the structure MeasHISTInfo  collected at the end of the FIFO measurement
 when FIDA , FILDA or MCS curves were created.
 It contains additionally information for MCS curve.
 The information is valid only for data blocks of type FIFO_DATA or FIFO_DATA_FROM_FILE

The structures are shown below.

*/


typedef struct {
  unsigned short  status;  // last SPC_test_state return value ( status )
  unsigned short  flags;   // scan clocks bits 2-0( frame, line, pixel),
              //  bit  3   - user break occured
              //      bits 4-8 for FIFO Image mode
              //  bit  4   - FIFO overrun happened
              //  bit  5   - FIFO read up to frame where collection timer expired
              //  bits 7-6 - wait_for_end_frame = 01 - wait for end of frame,
              //                                = 10 - end of frame was found
              //  bit  8   - measurement was running ( 1st Frame&Line present)
              //  bits 9   - wait_for_marker 3 to start the measurement
              //  bit  15  - reading rates was on

  float  stop_time;        // time from start to  - disarm ( simple measurement )
                           //    - or to the end of the cycle (for complex measurement )
  int    cur_step;         // current step  ( if multi-step measurement )
  int    cur_cycle;        // current cycle (accumulation cycle in FLOW mode ) -
                           //  ( if multi-cycle measurement )
  int    cur_page;         // current measured page
  float  min_sync_rate;    // minimum rates during the measurement
  float  min_cfd_rate;     //   ( -1.0 - not set )
  float  min_tac_rate;
  float  min_adc_rate;
  float  max_sync_rate;    // maximum rates during the measurement
  float  max_cfd_rate;     //   ( -1.0 - not set )
  float  max_tac_rate;
  float  max_adc_rate;
  char   stop_info_extension_used;    // MeasureStopInfoExt used
  char   reserved1[3];
  }MeasStopInfo;           // information collected when measurement is finished



typedef struct {
  float  min_ch4_rate;     //   ( -1.0 - not set )
  float  max_ch4_rate;     //
  float  min_ch5_rate;     //
  float  max_ch5_rate;     //
  float  reserve[8];
                           //  ch1 rate = cfd_rate, ch2 rate = tac_rate, ch3 rate = adc_rate, ch6 rate = sync_rate
  }MeasStopInfoExt;           // extension of MeasStopInfo - keeps additional info for module SPC_QCx06


typedef struct {
  unsigned short  chan;               // routing channel number, bits 6-4 = input channel for QC-x0x modules
  unsigned short  fcs_decay_calc;     // bit 0 = 1 - decay curve calculated
                                      // bit 1 = 1 - fcs   curve calculated
                                      // bit 2 = 1 - FIDA  curve calculated
                                      // bit 3 = 1 - FILDA curve calculated
                                      // bit 4 = 1 - MCS curve calculated
                                      // bit 5 = 1 - PS FLIM  image calculated
                                      // bit 6 = 1 - MCS_TA curve calculated
                                      // bit 7 = 1 - MCS FLIM image calculated
                                      // bit 8 = 1 - INTENSITY image calculated
                                      // bit 9 = 1 - WF (Wide-Field) FLIM image calculated
  unsigned int    mt_resol;           // macro time clock in 0.1 ns ( 1fs for DPC-230) units
  float           cortime;            // correlation time [ms]
  unsigned int    calc_photons;       //  total no of photons in decay histogram
  int             fcs_points;         // no of FCS values
  float           end_time;           // macro time of the last photon
  unsigned short  overruns;           // no of Fifo overruns
                              //   when > 0  fcs curve & end_time are not valid
  unsigned short  fcs_type;   // 0 - linear FCS with log binning ( 100 bins/log )
                              // when bit 15 = 1 ( 0x8000 ) - Multi-Tau FCS
                              //           where bits 14-0 = ktau parameter
  unsigned short  cross_chan;         // cross FCS routing channel number, bits 6-4 = input channel for QC-x0x modules
                   //   when chan = cross_chan and mod == cross_mod - Auto FCS
                   //        otherwise - Cross FCS
  unsigned short  mod;                // module number
  unsigned short  cross_mod;          // cross FCS module number
  unsigned int    cross_mt_resol;     // macro time clock of cross FCS module in 0.1 ns units
  }MeasFCSInfo;   // information collected when FIFO measurement is finished



typedef struct {
  float           fida_time;          // interval time [ms] for FIDA histogram
  float           filda_time;         // interval time [ms] for FILDA histogram
  int             fida_points;        // no of FIDA values
                                      //    or current frame number ( fifo_image)
                                      //    or no of not matched events in T channel (WF FLIM mode)
  int             filda_points;       // no of FILDA values
                                      //    or current line  number ( fifo_image)
                                      //    or no of not matched events in X channel (WF FLIM mode)
  float           mcs_time;           // interval time [ms] for MCS histogram
  int             mcs_points;         // no of MCS values
                                      //    or current pixel number ( fifo_image)
                                      //    or no of not matched events in Y channel (WF FLIM mode)
  unsigned int    cross_calc_phot;    //  no of calculated photons from cross_channel
                                      //    for Cross FCS histogram
                                      //    or no of calculated photons in X channel (WF FLIM mode)
  unsigned short  mcsta_points;       // no of MCS_TA values
  unsigned short  mcsta_flags;        // MCS_TA flags   bit 0 = 1 - use 'invalid' photons,
                                      //      bit 1-2  =  marker no used as trigger
  unsigned int    mcsta_tpp;          // MCS_TA Time per point  in Macro Time units
                                      // time per point[s] = mcsta_tpp * mt_resol( from MeasFCSInfo)
  unsigned int    calc_markers;       // no of calculated markers for MCS_TA
  unsigned int    fcs_calc_phot;      //  no of calculated photons for FCS histogram
                                      //    or no of calculated photons in Y channel (WF FLIM mode)
  unsigned int    reserved3;
  }MeasHISTInfo; // extension of MeasFCSInfo for histograms ( FIDA, FILDA, MCS, FCS, MCS_TA )


typedef struct {  // keep always 64 bytes size for compatibility
                  // subsequent 7 values are valid only in fifo_image mode
  float   first_frame_time;  //   macro time of the 1st frame marker
  float   frame_time;        //   time between first two frame markers
  float   line_time;         //   time between first two line markers ( in the 1st frame)
  float   pixel_time;        //   time between first two pixel markers ( in the 1st frame&line)
  short   scan_type;         // 0 - Unidir, 1 - Bidir
  short   skip_2nd_line_clk; // Fifo Image ( some microscopes delevers Line clock
                             //             at the beg. and at the end of the line )
  unsigned int    right_border;     // for Bidir scanning
  char    info[40];          //   not used yet
  }MeasHISTInfoExt; // extension of MeasHISTInfo for additional histograms info



typedef struct {
  unsigned int   DCU_in_use;  // bits 0..3 = 1 when DCU module M1..M4 was in use
                              // bits 4..8, 12..16, 18..22, 26..30- bit = 1
                              //   when connector 1..5 of module M1..M4 outputs were enabled
  char     dcu_ser_no[4][16];    //  serial number of used DCU modules
  char     scope_name[32];    // name of connected microscope
  char     lens_name[64];     // name of lens used with connected microscope
  unsigned int   SIS_in_use;  // bits 0..3 = 1 when SIS module M1..M4 was in use
                              // bits 4..5, 8..9, 12..13, 16..17- bit = 1
                              //   when switch A..B of module M1..M4 outputs were enabled
  char     sis_ser_no[4][16]; //  serial number of used SIS modules
  char     gvd_ser_no[16];    //  serial number of used GVD module,
                              //         = 0 - GVD module was not used
  float    zoom_factor;       //  scanner zoom factor
  float    FOV_at_zoom_1;     //  scanner FOV at zoom = 1 , in µm
  short    scope_connected;    // microscope mode = 0 - not connected, 1 - simulation w/o MTBAPI,
                              //                 2 - simulation,    3 - hardware,
  float    lens_magnifier;    //  lens magnifier
  float    image_size;        //  used image size, in µm
                              //  0 - not valid
                              //  = FOV_at_zoom_1 / zoom_factor / lens_magnifier
                              //    if values available and image_size_source = 0
  float    tdc_offset[4];     //  TDC offset for 4 inputs of QC-x0x modules
  unsigned long tdc_control;  //  TDC control bits
  union {                     //  !!! potential problem for LabView  -
                              //     should be 4 bytes ( start of tdc_offset106 is not adjusted to 4 )
    unsigned char scale_bar;        // image scale bar : bit 0 = 0 - off, 1 - on ( visible)
    struct {                        //  bits 5-4 system type = 00 - not defined,
      unsigned char sbEnable : 1;   //    01 - GVD on motorized Axio,
      unsigned char : 3;            //    10 - GVD on manual Axio - currently not available
      unsigned char sbSysType : 2;  //    11 - LSM980, no GVD
      };
    };
  float    tdc_offset106[2];     //  TDC offset for inputs 5 & 6 of QC-x06 modules
  float    cfd_th5_tdc;          //  for QC-x06 modules, threshold of CFD5 -500 ..0 mV
  float    cfd_th6_tdc;          //  for QC-x06 modules, threshold of CFD6 -500 ..0 mV
  float    cfd_zc5_tdc;          //  for QC-x06 modules, Zero Cross level of CFD5 -96 ..96 mV
  float    cfd_zc6_tdc;          //  for QC-x06 modules, Zero Cross level of CFD6 -96 ..96 mV
  MeasStopInfoExt  StopInfoExt;  // extension of MeasStopInfo - keeps additional info for module SPC_QC-x06
  char     mwl_ser_no[32];       //  serial number of used MWL,
                                 //         = 0 - MWL was not used
  int      mwl_grating;          //  current grating value
  float    mwl_wav;              //  current centre wavelength [nm] of monochromator
  char     reserve[1137];        // keep always total size = 1536B
  }MeasureInfoExt;        // extension of MeasureInfo for additional info



typedef struct _MeasureInfo {
  char     time[9];        /* time of creation */
  char     date[11];       /* date of creation */
  char     mod_ser_no[16]; /* serial number of the module */
  short    meas_mode;
  float    cfd_ll;    // CFD1 Threshold for DPC-230 & QC-x0x modules
  float    cfd_lh;    // CFD2 Threshold for DPC-230 & QC-x0x modules
  float    cfd_zc;    // CFD3 Threshold for DPC-230 & QC-x0x modules
  float    cfd_hf;    // CFD4 Threshold for DPC-230, CFD3 ZC Level  for QC-x0x modules
  float    syn_zc;    // CFD1 ZC Level  for DPC-230, CFD4 ZC Level  for QC-x0x modules
  short    syn_fd;
  float    syn_hf;    // CFD2 ZC Level  for DPC-230 & QC-x0x modules
  float    tac_r;
  short    tac_g;
  float    tac_of;
  float    tac_ll;
  float    tac_lh;    // CFD4 ZC Level  for DPC-230, CFD1 ZC Level  for QC-x0x modules
  short    adc_re;    // value 0 means 65536, Points No for DPC-230 & QC-x0x modules
  short    eal_de;
  short    ncx;
  short    ncy;
  unsigned short  page;
  float    col_t;
  float    rep_t;
  short    stopt;
  char     overfl;
  short    use_motor;
  unsigned short    steps;
  float    offset;
  short    dither;
  short    incr;
  short    mem_bank;
  char     mod_type[16];   /* module type */
  float    syn_th;    // CFD3 ZC Level  for DPC-230, CFD4 Threshold  for QC-x0x modules
  short    dead_time_comp;
  short    polarity_l;   //  2 = disabled line markers
  short    polarity_f;
  short    polarity_p;
  short    linediv;      // line predivider = 2 ** ( linediv)
  short    accumulate;
  int      flbck_y;
  int      flbck_x;
  int      bord_u;
  int      bord_l;
  float    pix_time;
  short    pix_clk;
  short    trigger;
  int      scan_x;
  int      scan_y;
  int      scan_rx;
  int      scan_ry;
  short    fifo_typ;
  int      epx_div;
  unsigned short  mod_type_code;
  unsigned short  mod_fpga_ver;    // new in v.8.4
  float    overflow_corr_factor;
  int      adc_zoom;
  int      cycles;        //  cycles ( accumulation cycles in FLOW mode )
  MeasStopInfo StopInfo;
  MeasFCSInfo  FCSInfo;   // valid only for FIFO meas
  int      image_x;       // 4 subsequent fields valid only for Camera mode
  int      image_y;       //     or FIFO_IMAGE mode
  int      image_rx;
  int      image_ry;
  short    xy_gain;       // gain for XY ADCs ( SPC930 )
  short    dig_flags; // SP_MST_CLK parameter bits 0-7 - digital flags :
                      //   bit 0 - use or not  Master Clock (SPC140 multi-module )
                      //   bit 1 - Continuous Flow On/Off for scan modes in SPC150
                      //   bit 2 - time(X) axis of decay curves reversed by software(1), or not (0)
  short    adc_de;        // ADC sample delay ( SPC-930 )
  short    det_type;      // detector type ( SPC-930 in camera mode )
  short    x_axis;        // X axis representation ( SPC-930 )
  MeasHISTInfo  HISTInfo; // extension of FCSInfo, valid only for FIFO meas
  MeasHISTInfoExt  HISTInfoExt; // extension of HSTInfo, valid only for FIFO meas
  float    sync_delay;    // Sync Delay [ns] when using BH SyncDel USB box
  unsigned short sdel_ser_no;  // serial number of Sync Delay box,
                               //         = 0 - SyncDelay box was not used
  char     sdel_input;    // active input of SyncDelay box, 0 - IN 1, 1 - IN 2
  char     mosaic_ctrl;   // bit 0 - mosaic imaging was used,
                          // bit 1 - mosaic type ( 00 - sequence of frames, 01 - rout. channels)
                          // bit 2-3 - mosaic restart type ( 0 - no, 1 - Marker 3, 2 - Ext. trigger
                          // bit 4-5 - mosaic type extension(
                          //               010 - sequence of Z planes of Axio Observer.Z1)
                          //               011 - sequence of frames with motor stage )
  unsigned char     mosaic_x;      //  no of mosaic elements in X dir.  (lower 8 bits )
  unsigned char     mosaic_y;      //  no of mosaic elements in Y dir.  (lower 8 bits )
  short    frames_per_el; //  frames per mosaic element 1 .. 32767
  short    chan_per_el;   //  routing channels per mosaic element 1 .. 256
  int      mosaic_cycles_done;  // number of mosaic accumulation cycles done
  unsigned short mla_ser_no;  // serial number of MLA4 device,
                              //         = 0 - MLA4 device was not used
  unsigned char  DCC_in_use;  // bits 0..3 = 1 when DCC module M1..M4 was in use
                              // bits 4..7 = 1 when DCC module M1..M4 outputs were enabled
  char     dcc_ser_no[12];    //  serial number of used DCC module
  unsigned short TiSaLas_status;  //  laser status :bit 0 - used or not, b1 - present,
                   // b2 - on/off, b3 - shutter, b4 - warm up, b5 - pulsing,
                   // b6 - error, b7-10 - laser type 0-15, 0 = unknown
  unsigned short TiSaLas_wav;  //  laser wavelength in nm
  unsigned char  AOM_status;   //  AOM status :bit 0 - used or not, b1 - present,
                               //              b2 - modulation, b3 - remote, b4 - error
  unsigned char  AOM_power;    //  AOM power in %  0 - 100
  char     ddg_ser_no[8];    //  serial number of used DDG module,
                              //         = 0 - DDG module was not used
  int      prior_ser_no;      // serial number of used Prior device, 0 - not used
  unsigned char     mosaic_x_hi;      //  no of mosaic elements in X dir.  (higher 8 bits )
  unsigned char     mosaic_y_hi;      //  no of mosaic elements in Y dir.  (higher 8 bits )
  char     reserve[11];       // total size of old MeasureInfo = 512 bytes
  char     extension_used;    // MeasureInfoExt used
  MeasureInfoExt  minfo_ext;  // extension structure - size fixed to 1536 bytes
                              // new total size of MeasureInfo = 2048 bytes
  }MeasureInfo;



/*
   5- Data Blocks

With the software version 7.0  the data block header was changed to make possible
  a higher number of data blocks and a variable block size.
Each data block can now contain a 'Data Set' i.e. the data of several curves
  which were obtained in one measurement.
The number and the location of  the data blocks is contained in the file header
  at the beginning of the data file.
The length of the block is contained in the block header.
The data block header contains the data block number,
  the offset of the data block from the beginning of the file,
  the offset to the next data block,
  an information about the data in the block (measured block, block loaded from file, etc.),
  and a reference to the corresponding measurement description block

The data block header structure is shown below.
*/

// The data block header structure for bhfile_header.revision < 15

/*
typedef struct {
   short          block_no;   // number of the block in the file
                         // valid only  when in 0 .. 0x7ffe range, otherwise use lblock_no field
                         // obsolete now, lblock_no contains full block no information
   long           data_offs;       // offset of the data block
   long           next_block_offs; // offset to the data block header of the next data block
   unsigned short block_type;      // see block_type defines below
   short          meas_desc_block_no; // Number of the measurement description block
                                      //    corresponding to this data block
   unsigned long  lblock_no;       // long block_no - see remarks below
   unsigned long  block_length;    // block( set ) length ( not compressed ) in bytes
   }BHFileBlockHeader;
*/

// The data block header structure for bhfile_header.revision = 15 - current state
typedef struct _bhfile_block_header {
   unsigned char  data_offs_ext;        // extension of data_offs field - address bits 32-39
   unsigned char  next_block_offs_ext;  // extension of next_block_offs field - address bits 32-39
   unsigned long  data_offs;       // offset of the block's data, bits 0-31
   unsigned long  next_block_offs; // offset to the data block header of the next data block, bits 0-31
   unsigned short block_type;      // see block_type defines above
   short          meas_desc_block_no;
   unsigned long  lblock_no;       // long block_no - see remarks below
   unsigned long  block_length;    // block( set ) length ( not compressed ) in bytes up to 2GB
   }BHFileBlockHeader;


//    all offsets are given from beginning of the file
//    ( MSB bit is 31, LSB bit is 0 )
//    The field block_no is obsolete
//    The field 'lblock_no' contains the data block / data set number in the bits 0 to 23
//          and the module number (0 to 3) in the bits 24 to 25.
//
//    Block numbering start always from 1, so before calculating position in SPC memory
//         ( or extracting FCS auto/cross rout. channels ) it should be at first decremented by 1
//
//    for data set ( PAGE_BLOCK )
//           bits 0 - 23 represent number of the 1st block (decay curve) of the page
//
//    for FIFO_DATA
//         bits 23 - 20   = block contents - the same as bits 4-7 in block_type
//                          ( starting from software version 8.3 )
//           for FIDA_BLOCK , FILDA_BLOCK, MCS_BLOCK, MCSTA_BLOCK and IMG_BLOCK
//             bits 7 - 0  represent routing channel number
//           for FCS_BLOCK bits 17 - 0  contain routing channel and module
//                                      for auto and cross FCS
//             bits 17 - 16   = Cross FCS module number
//             bits 15 - 8    = rout. channel number
//             bits  7 - 0    = Cross FCS rout. channel number
//             when chan = cross_chan and mod == cross_mod - Auto FCS,
//                           otherwise - Cross FCS
//    for MOM_DATA
//         bits 23 - 20   = block(set) contents - the same as bits 4-7 in block_type
//         bits 19 -  0   = number of the data set


//////////
// defines for block type field in BHFileBlockHeader
/////////

// bits 0 - 3  in block_type define mode of creation

#define NOT_USED              0
#define MEAS_DATA             1   // normal measurement modes
#define FLOW_DATA             2   // continuos flow measurement ( BIFL )
#define MEAS_DATA_FROM_FILE   3
#define CALC_DATA             4   // calculated data
#define SIM_DATA              5   // simulated data
#define FIFO_DATA             8   // FIFO mode data
#define FIFO_DATA_FROM_FILE   9   // FIFO mode data
#define MOM_DATA             10   // moments mode data
#define MOM_DATA_FROM_FILE   11

// bits 4 - 7  in block_type define block contents

#define DECAY_BLOCK    0x0     // one decay curve
#define PAGE_BLOCK     0x10    // set of decay curves = measured page
#define FCS_BLOCK      0x20    // FCS histogram curve
#define FIDA_BLOCK     0x30    // FIDA histogram curve
#define FILDA_BLOCK    0x40    // FILDA histogram curve
#define MCS_BLOCK      0x50    // MCS histogram curve
#define IMG_BLOCK      0x60    // fifo image - set of curves = PS FLIM
#define MCSTA_BLOCK    0x70    // MCS Triggered Accumulation histogram curve
#define IMG_MCS_BLOCK  0x80    // fifo image - set of curves = MCS FLIM
#define MOM_BLOCK      0x90    // moments mode - set of moments data frames
#define IMG_INT_BLOCK  0xa0    // fifo image - INTENSITY ( only for SPC-16x/18x)
#define IMG_WF_BLOCK   0xb0    // fifo image - Wide-Field FLIM


// bits 8 - 11 in block_type define data type

#define DATA_USHORT           0x0     // 16-bit unsigned short
                                      //   (it was the only type up to 20.08.04)
#define DATA_ULONG            0x100   // 32-bit unsigned long, for FIFO decay curves
#define DATA_DBL              0x200   // 64-bit double, for histogram data blocks

// bit 12 = 1 -   block contains compressed data
#define DATA_ZIPPED           0x1000  // data block is compressed
// bit 14 = 1 -   block is compressed with LZ4 DLL
#define DATA_ZIP_LZ4          0x4000  //  data block is compressed with LZ4 DLL
// bit 13 = 1 -   block contains coincidence data
#define COI_BLOCK             0x2000  // coincidence data block ( SPC-QCx0x modules in Fifo Absolute Time mode)

// bits 14- 15 in block_type are not yet defined, reserved for future use

      // defines for page data sets
#define MEAS_PAGE             0x11
#define FLOW_PAGE             0x12
#define MEAS_PAGE_FROM_FILE   0x13
#define CALC_PAGE             0x14
#define SIM_PAGE              0x15



/*
 Data of the block
 for all blocks except FIFO_DATA ( FIFO_DATA_FROM_FILE ) ( bits 0 - 3  in block_type )

The data of the set specified by the block header is stored as shown below.
It follows directly after the data block header:
  unsigned short  curvepoint[0][0]
  unsigned short  curvepoint[0] [1]
  . .
  . .             curvepoint[0] [adc_re -1]
  unsigned short  curvepoint[1][0]
  unsigned short  curvepoint[1] [1]
  . .
  .               curvepoint[1] [adc_re -1]
  .
  unsigned short  curvepoint[n][0]
  unsigned short  curvepoint[n] [1]
  . .
  .               curvepoint[n] [adc_re -1]


The photon numbers in each curve point are unsigned short integers,
  i.e. values from 0 to 65,535.
The number of curves in the data set depends on the measurement parameters,
  e.g. measurement mode, no of routing bits etc.
The number of curves in the block is equal to
        'block_length' (from the block header) divided by
        adc_resolution (from the corresponding measurement description block).
*/

/*

 Data of the block
 only for blocks FIFO_DATA ( FIFO_DATA_FROM_FILE ) ( bits 0 - 3  in block_type )
 for files created with software version > 8.2

 Depending on bits 4 - 7  in block_type data block of type FIFO_DATA ( FIFO_DATA_FROM_FILE )
 can contain one decay curve or FCS, FIDA, FILDA histograms data.
 For IMG_BLOCK it contains set of decay curves representing the image.
 Decay curve contains adc_re points which are unsigned long integers ( 32-bit ).

 FCS histogram data (FCS_BLOCK) contain results of FCS (auto or cross ) calculation for photons
    collected in one routing channel.
    Number of FCS curve points is defined in coresponding measurement description
    block in the field FCSInfo.fcs_points.
    FCS calculation uses logaritmic binning therefore both time ( x ) and data ( y )
      values are required.
    At first FCSInfo.fcs_points values of FCS time in [µs] ( x scale ) are  given
      and then FCSInfo.fcs_points values of FCS curve ( y scale ).
    Each FCS time and data point is a 8 byte double value.

 FIDA histogram data (FIDA_BLOCK) contains results of FIDA calculation for photons
    collected in one routing channel.
    Number of FIDA curve points is defined in coresponding measurement description
    block in the field HISTInfo.fida_points.
    Each FIDA data point is a 8 byte double value.

 FILDA histogram data (FILDA_BLOCK) contains results of FILDA calculation for photons
    collected in one routing channel.
    Number of FILDA curve points is defined in coresponding measurement description
    block in the field HISTInfo.filda_points.
    Each FILDA data point is a 8 byte double value.

 only for files created with software version > 8.3
 MCS histogram data (MCS_BLOCK) contains results of MCS calculation for photons
    collected in one routing channel.
    Number of MCS curve points is defined in coresponding measurement description
    block in the field HISTInfo.mcs_points.
    Each MCS data point is a 4 byte unsigned long integer value.

 IMG_BLOCK data contains set of decay curves representing the image built from photons
    collected in one routing channel.
    Curve length is equal to adc_re from the corresponding measurement description block
    The number of curves in the block is equal to
        'block_length' (from the block header) divided by
         adc resolution (adc_re from the corresponding measurement description block)
         and is normally equal to image_x * image_y
    The photon numbers in each curve point are unsigned short integers,
      i.e. values from 0 to 65,535.


 only for files created with software version >= 9.0

 MCSTA histogram data (MCSTA_BLOCK) contains results of MCS Triggered Accumulation calculation
   for photons collected in one routing channel.
   The histogram contains microsecond decay curve with reference pulse connected to marker inputs.

   Number of MCS curve points is defined in coresponding measurement description
    block in the field HISTInfo.mcsta_points.
   Each MCSTA data point is a 4 byte unsigned long integer value.

 IMG_MCS_BLOCK data contains set of microsecond decay curves representing the image built from photons
    collected in one routing channel in Fifo Image mode with activated MCS FLIM option.
    Curve length is equal to HISTInfo.mcsta_points from the corresponding measurement description block
    The number of curves in the block is equal to
        'block_length' (from the block header) divided by
          mcsta_points (HISTInfo.mcsta_points from the corresponding measurement description block)
         and is normally equal to image_x * image_y
    The photon numbers in each curve point are unsigned short integers,
      i.e. values from 0 to 65,535.


  /////////////////////////////////////////////////////
 only for files created with software v. 8.1 - 8.2
 Data of the block
 only for blocks FIFO_DATA ( FIFO_DATA_FROM_FILE ) ( bits 0 - 3  in block_type )

  Data block of type FIFO_DATA ( FIFO_DATA_FROM_FILE ) contains one decay curve
   as described above ( 16-bit unsigned short integers).
   After the last curvepoint FCS curve is stored which contains results of
    FCS (auto or cross ) calculation for photons collected in one routing channel.
  Number of FCS curve points is defined in coresponding measurement description
   block in the field FCSInfo.fcs_points.
  FCS calculation uses logaritmic binning therefore both time ( x ) and data ( y )
   values are required.
  At first FCSInfo.fcs_points values of FCS time in [µs] ( x scale ) are  given
    and then FCSInfo.fcs_points values of FCS curve ( y scale ).
  Each FCS time and data point is a 8 byte double value.
  /////////////////////////////////////////////////////

*/



/////////////////////////////////////////////////////////////////////////
////////////         .spc  files structure       ////////////////////////
/////////////////////////////////////////////////////////////////////////

/*

Measurements in the FIFO mode generate one or several a photon data file files
  ( with extension .spc ) containing the micro times, the macro times,
  and the detector channel numbers of the individual photons.

  Moreover, a FIFO measurement generates a setup file ( .set) that contains
    the hardware and software setup parameters used. Structure of the setup file
    is described above.
*/


////////////////////////////////////////////////////////////////////////////
///    FIFO Data Files (SPC-600/630, 4096 Channel Mode, 48-bit photon frame)
////////////////////////////////////////////////////////////////////////////

/*

The information about the subsequent photons is stored one after another
 in the measurement data file.
 For each photon 6 bytes are used.
 The structure of these data is shown in the table below.

           Bit 7   Bit 6   Bit 5   Bit 4   Bit 3   Bit 2   Bit 1   Bit 0
------------------------------------------------------------------------
Byte 0    ADC[7]  ADC[6]  ADC[5]  ADC[4]  ADC[3]  ADC[2]  ADC[1]  ADC[0]
Byte 1    0       GAP     MTOV   INVALID  ADC[11] ADC[10] ADC[9]  ADC[8]
Byte 2    MT[23]  MT[22]  MT[21]  MT[20]  MT[19]  MT[18]  MT[17]  MT[16]
Byte 3    R[7]    R[6]    R[5]    R[4]    R[3]    R[2]    R[1]    R[0]
Byte 4    MT[7]   MT[6]   MT[5]   MT[4]   MT[3]   MT[2]   MT[1]   MT[0]
Byte 5    MT[15]  MT[14]  MT[13]  MT[12]  MT[11]  MT[10]  MT[9]   MT[8]

ADC[11:0]    ADC Data (reversed Micro Time)
             Due to reversed start-stop acquisition mode used in all SPC modules
             real Micro Time = 4095 - ADC
R [7:0]      Routing Signals (inverted)
MT[23:0]     Macro Time [in 50ns intervals]
GAP          1 = Possible recording gap due to FIFO Full.
             There may be (and most likely is) a gap in the re-cording preceding
               this photon.
MTOV         1 = Macro Timer Overflow.
             Since the capacity of the macro timer is limited to 24 bit
             it will overflow every 2**24 × 50 ns.
             The software which processes the data file has to add these 2**24 × 50 ns
             to its internal macro time value on each MTOV =1.
INVALID      1 = Data Invalid. All data for this photon except the MTOV bit is invalid.
             The INVALID bit is set if the 'Count Enable' bit at the SPC routing connector was '0',
              i.e. if a router is connected and there is no valid routing information
              for this photon.

The first photon frame (6 bytes) in the .spc file is added by the software and contains information
about the Macro Time clock and the number of routing channels used during the measurement:
  byte 1 : bit 4 = 1 ( Invalid photon ),  bits 3-0 = number of routing bits
  byte 2 and 3  = macro time clock in 0.1 ns units ( for 50ns value 500 is set )
  byte 5 and 6  = 0

when programming, the most convenient way is to treat  the  photon frame as 3 16-bit words

1st data word - Byte 1 ( bits 15-8 ), Byte 0 ( bits 7-0 )
2nd data word - Byte 3 ( bits 15-8 ), Byte 2 ( bits 7-0 )
3rd data word - Byte 5 ( bits 15-8 ), Byte 5 ( bits 7-0 )

*/

 //  0x1000000 * 50e-9, macro time overflow  for SPC-6x0 in 48-bit mode
#define OV_TIME48      0.8388608
   // masks for 1st word,  SPC-6x0 in 48-bit mode
#define MTOV48        0x2000     // Macro Time overflow
#define OVRUN48       0x4000     // Fifo overrun, recording gap
#define INVALID48     0x1000     // Invalid entry

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////
///    FIFO Data Files (SPC-600/630, 256 Channel Mode, 32-bit photon frame)
/////////////////////////////////////////////////////////////////////////

/*

The information about the subsequent photons is stored one after another
 in the measurement data file. For each photon 4 bytes are used.
 The structure of these data is shown in the table below.

           Bit 7    Bit 6    Bit 5    Bit 4    Bit 3    Bit 2    Bit 1    Bit 0
-------------------------------------------------------------------------------
Byte 0     ADC[7]   ADC[6]   ADC[5]   ADC[4]   ADC[3]   ADC[2]   ADC[1]   ADC[0]
Byte 1     MT[7]    MT[6]    MT[5]    MT[4]    MT[3]    MT[2]    MT[1]    MT[0]
Byte 2     MT[15]   MT[14]   MT[13]   MT[12]   MT[11]   MT[10]   MT[9]    MT[8]
Byte 3     INVALID  MTOV     GAP      0        R[2]     R[1]     R[0]     MT[16]

ADC[7:0]     ADC Data (reversed Micro Time)
             Due to reversed start-stop acquisition mode used in all SPC modules
             real Micro Time = 255 - ADC
R [2:0]      Routing Signals (inverted)
MT[16:0]     Macro Time [in 50ns intervals]
GAP          1 = Possible recording gap due to FIFO Full.
             There may be (and most likely is) a gap in the re-cording preceding
               this photon.
MTOV         1 = Macro Timer Overflow.
             Since the capacity of the macro timer is limited to 17 bit
             it will overflow every 2**17 × 50 ns.
             The software which processes the data file has to add these 2**17 × 50 ns
             to its internal macro time value on each MTOV = 1.
INVALID      1 = Data Invalid. All data for this photon except the MTOV bit is invalid.
             The INVALID bit is set if the 'Count Enable' bit at the SPC routing connector was '0',
              i.e. if a router is connected and there is no valid routing information
              for this photon.

The INVALID bit is used to mark photons for which no correct detector channel number,
   micro time, or macro time could be determined.
   This happens if several detectors in a multi-detector system detected photons within
   the response time of the routing electronics, the sum of the TAC output and the dither signal
   was outside the conversion range of the ADC, or in the (very unlikely) case
   that a photon appears within the generation of a macro time overflo entry (see below).

The first photon frame (4 bytes) in the .spc file is added by the software and
  contains information about the Macro Time clock and the number of routing channels
  used during the measurement:
  byte 0,1,2    = macro time clock in 0.1 ns units ( for 50ns value 500 is set )
  byte 3 : bit 7 = 1 ( Invalid photon ),  bits 3-6 = number of routing bits,
           bits 0-2 reserved


Macro Time Overflows
Due to the high macro time resolution and the limited number of macro time bits
  a macro time overflow occurs every 6.5536 ms.
  In most cases a photon will be recorded within this time.
  The ocurrance of  a single macro time overflow is then marked by the 'MTOV' bit of the next valid photon.
  However, for low count rates it may happen that several macro time overflows occur
  between subsequent photons.
  In this case an entry in the FIFO file is generated that contains the number of overflows
  since the last photon was recorded.
  This entry is marked by 'MTOV = 1' and 'INVALID = 1'.
  The structure of  this entry is shown below.

           Bit 7    Bit 6    Bit 5    Bit 4    Bit 3    Bit 2    Bit 1    Bit 0
-------------------------------------------------------------------------------
Byte 0    CNT[7]   CNT[6]   CNT[5]   CNT[4]   CNT[3]   CNT[2]   CNT[1]   CNT[0]
Byte 1    CNT[15]  CNT[14]  CNT[13]  CNT[12]  CNT[11]  CNT[10]  CNT[9]   CNT[8]
Byte 2    CNT[23]  CNT[23]  CNT[21]  CNT[20]  CNT[19]  CNT[18]  CNT[17]  CNT[16]
Byte 3    INVALID  MTOV     0        0        CNT[27]  CNT[26]  CNT[25]  CNT[24]

CNT[27:0] Number of macro time overflows which occurred without recording photons

when programming, the photon frame is treated as one 32-bit word

*/

 //  0x20000 * 50e-9, macro time overflow  for SPC-6x0 in 32-bit mode
#define OV_TIME6_32    0.0065536

  // masks for different parts of the frame
#define INVALID6_32        0x80000000   // Invalid
#define MTOV6_32           0x40000000   // Macro time overflow
#define INVALID_MTOV6_32   0xc0000000   // Invalid + Macro time overflow
#define OVRUN6_32          0x20000000   // Fifo overrun, recording gap
#define ROUT6_32           0x07000000   // Routing signals( inverted )
#define MT6_32             0x01ffff00   // Macro time
#define ADC6_32            0xff         // ADC value

  // masks for the 1st frame in .spc file
#define RB_NO6_32          0x78000000   // routing bits number used during measurement
#define MT_CLK6_32         0x00ffffff   // macro time clock in 0.1 ns units
#define R_FILE6_32         0x04000000   // file with raw data ( diagnostic mode only )


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////
///    FIFO Data Files (SPC-13x, SPC-140, SPC-15x, SPC-830, SPC-16x, SPC-18x )
//////////////////////////////////////////////////////////////

/*

The information about the subsequent photons is stored one after another
 in the measurement data file. For each photon 4 bytes are used.
 The structure of these data is shown in the table below.

           Bit 7    Bit 6    Bit 5    Bit 4    Bit 3    Bit 2    Bit 1    Bit 0
-------------------------------------------------------------------------------
Byte 0     MT[7]    MT[6]    MT[5]    MT[4]    MT[3]    MT[2]    MT[1]    MT[0]
Byte 1     R[3]     R[2]     R[1]     R[0]     MT[11]   MT[10]   MT[9]    MT[8]
Byte 2     ADC[7]   ADC[6]   ADC[5]   ADC[4]   ADC[3]   ADC[2]   ADC[1]   ADC[0]
Byte 3     INVALID  MTOV     GAP      0        ADC[11]  ADC[10]  ADC[9]   ADC[8]

ADC[11:0]  ADC Data (reversed Micro Time)
           Due to reversed start-stop acquisition mode used in all SPC modules
             real Micro Time = 4095 - ADC
R [3:0]    Routing Signals (inverted).
           Bit 3 is not available for early SPC-130/134 modules
MT[16:0]   Macro Time (50 ns intervals for internal Macro time oscillator,
                        for Sync clock see below)
GAP        1 = Possible recording gap due to FIFO Full.
           There may be (and most likely is) a gap in the re-cording preceding
             this photon.
MTOV       1 = Macro Timer Overflow.
           Since the capacity of the macro timer is limited to 12 bit
           it will overflow every 2**12th macro time period.
           For the internal macro time clock of 50 ns overflows occur every 204.8 µs.
           For synchronisation of the macro-time clock with the SYNC signal
             this interval may even be shorter.
           Therefore there may even be multiple overflows without any photons detected (see below).
           The software which processes the data file has to add the times
             between overflows to its internal macro time value on each MTOV =1
INVALID    1 = Data Invalid. All data for this photon except the MTOV bit is invalid.
           The INVALID bit is set if the 'Count Enable' bit at the SPC routing connector was '0',
            i.e. if a router is connected and there is no valid routing information
            for this photon.

The INVALID bit is used to mark photons for which no correct detector channel number,
   micro time, or macro time could be determined.
   This happens if several detectors in a multi-detector system detected photons within
   the response time of the routing electronics, the sum of the TAC output and the dither signal
   was outside the conversion range of the ADC, or in the (very unlikely) case
   that a photon appears within the generation of a macro time overflo entry (see below).

The first photon frame (4 bytes) in the .spc file is added by the software and contains information
about the Macro Time clock and the number of routing channels used during the measurement:
  byte 0,1,2    = macro time clock in 0.1 ns units ( for 50ns value 500 is set )
  byte 3 : bit 7 = 1 ( Invalid photon ),  bits 3-6 = number of routing bits,
           bit 2 = 1 file with raw data ( for diagnostic mode only,
                     such file contains pure data read from FIFO without special meaning
                       of the entries with 'MTOV = 1' and 'INVALID = 1' described below )
           bit 1 = 1 file with markers ( new SPC-140, SPC-15x/16X/18x, see below)
           bit 0 reserved

Internal Macro Time Clock is equal 50ns for all modules.
The exception is : SPC-140 with FPGA v. > B0, SPC-15x, SPC-16x and SPC-18x where it is equal 25 ns.

Macro Time Overflows
  Since the capacity of the macro timer is limited to 12 bit it will overflow
   at each 2**12-th macro time period.
  In most cases the overflow is marked by the 'MTOV' bit of the next valid photon.
  However, the internal macro time clock of 50 ns overflows occur every 204.8 µs.
  For synchronisation of the macro-time clock with the SYNC signal this interval may
    even be shorter.
  Therefore, it can happen that no valid photon is recorded between two subsequent
    macro time overflows.
  To enable the processing software to maintain a correct macro time for
    the rest of the measurement an entry in the measurement data file is written
    if several overflows occurred between two subsequent photons.
  This entry is marked by 'MTOV = 1' and 'INVALID = 1' and contains
   the number of macro time overflows which occurred since the last photon was recorded.

  The structure of  this entry is shown below.

           Bit 7    Bit 6    Bit 5    Bit 4    Bit 3    Bit 2    Bit 1    Bit 0
-------------------------------------------------------------------------------
Byte 0    CNT[7]   CNT[6]   CNT[5]   CNT[4]   CNT[3]   CNT[2]   CNT[1]   CNT[0]
Byte 1    CNT[15]  CNT[14]  CNT[13]  CNT[12]  CNT[11]  CNT[10]  CNT[9]   CNT[8]
Byte 2    CNT[23]  CNT[23]  CNT[21]  CNT[20]  CNT[19]  CNT[18]  CNT[17]  CNT[16]
Byte 3    INVALID  MTOV     0        0        CNT[27]  CNT[26]  CNT[25]  CNT[24]

CNT[27:0] Number of macro time overflows which occurred without recording photons

when programming, the photon frame is treated as one 32-bit word

*/

 //  0x1000 * 50e-9, macro time overflow with internal macro time clock
#define OV_TIME32      0.0002048

  // masks for different parts of the frame
#define INVALID32        0x80000000   // Invalid
#define MTOV32           0x40000000   // Macro time overflow
#define INVALID_MTOV32   0xc0000000   // Invalid + Macro time overflow
#define OVRUN32          0x20000000   // Fifo overrun, recording gap
#define ROUT32           0xf000       // Routing signals( inverted )
#define MT32             0xfff        // Macro time
#define ADC32            0x0fff0000   // ADC value


  // masks for the 1st frame in .spc file
#define RB_NO32          0x78000000   // routing bits number used during measurement
#define MT_CLK32         0x00ffffff   // macro time clock in 0.1 ns units
#define M_FILE32         0x02000000   // file with markers
#define R_FILE32         0x04000000   // file with raw data ( diagnostic mode only )


/////////////////////////////////////////////////////////////////////////
///  additional information for  FIFO Data Files with markers
/////////////////////////////////////////////////////////////////////////

/*

The newest SPC-140 ( FPGA v. >= B0 ), SPC-830 ( FPGA v. >= C0) SPC-150 and SPC-160 can work in
  FIFO_IMAGE mode and can collect marker entries.
Module SPC-130EM can collect marker entries but does not work in FIFO_IMAGE mode.

Internal macro time clock for these modules is equal 25 ns.

In this mode  not only photons are collected in the module's FIFO,
  but also markers entries.
 Marker entries are created from external Frame, Line and Pixel clocks pulses.
 Marker entries contain valid macro time information.
  for SPC-160 modules bits ADC[11:0] ( Byte 2 & 3 ) = INT[11:0] Intensity data =
         Number of photons detected by CFD between last two subsequent pixel markers
         This data is used in SPCM application to build intensity image
  for other modules ADC[11:0] ( Byte 2 )is not used ( do not care) ,
  R0-3 bits contain marker's type

Markers can be used also in normal FIFO mode.
Up to four user defined events can be recorded in this way together with normal photons.

To record markers in FIFO mode enable used markers' inputs in
   System Parameters->MoreParameters window when FIFO mode is chosen


.spc files with markers are recognized by flag M_FILE32 set in the 1st file frame.

The structure of the marker frame is shown in the table below.

           Bit 7    Bit 6    Bit 5    Bit 4    Bit 3    Bit 2    Bit 1    Bit 0
-------------------------------------------------------------------------------
Byte 0     MT[7]    MT[6]    MT[5]    MT[4]    MT[3]    MT[2]    MT[1]    MT[0]
Byte 1     M[3]     M[2]     M[1]     M[0]     MT[11]   MT[10]   MT[9]    MT[8]
Byte 2     x        x        x        x        x        x        x        x
Byte 3     1        MTOV     GAP      1        x        x        x        x

for SPC-160
Byte 2     INT[7]   INT[6]   INT[5]   INT[4]   INT[3]   INT[2]   INT[1]   INT[0]
Byte 3     1        MTOV     GAP      1        INT[11]  INT[10]  INT[9]   INT[8]

x - not used
M [3:0]   Marker's type:
          M[0] = 1   Marker0 = Pixel Clock Marker
          M[1] = 1   Marker1 = Line  Clock Marker
          M[2] = 1   Marker2 = Frame Clock Marker
          M[3] = 1   Marker3, not used in FIFO_IMAGE mode, user defined

It can happen that in the same frame more than 1 marker M[] bits are set.
It would mean that different external clock pulses appeared in the current
  macro time period
*/

  // masks for markers frames

#define MARKER0       0x90001000
#define MARKER1       0x90002000
#define MARKER2       0x90004000
#define MARKER3       0x90008000

#define P_MARK32      MARKER0   // pixel marker
#define L_MARK32      MARKER1   // line  marker
#define F_MARK32      MARKER2   // frame marker






//////////////////////////////////////////////////////////////
///    FIFO Data Files (DPC-230 )
//////////////////////////////////////////////////////////////
/*
           DPC-230 file format
A DPC-230 .spc file created by Becker & Hickl SPCM software consists of a sequence
  of 32 bit records.
The byte order shown is the byte order in the file.
-------------------------------------------------------------------------------
  Byte 0  1. Record = Descriptor  Bits [7:0]
  Byte 1                          Bits [15:8]
  Byte 2                          Bits [23:16]
  Byte 3                          Bits [31:24]
-------------------------------------------------------------------------------
  Byte 0  2. Record               Bits [7:0]
  Byte 1                          Bits [15:8]
  Byte 2                          Bits [23:16]
  Byte 3                          Bits [31:24]
-------------------------------------------------------------------------------

The first record is a descriptor which identifies DPC-230 files and defines the time unit.

Descriptor

  Bit 31  Bit 30  Bit 29  Bit 28  Bit 27  Bit 26  Bit 25  Bit 24
     1       1       0       0       0     RAW       0       1

  Bit 23  Bit 22  Bit 21  Bit 20  Bit 19  Bit 18  Bit 17  Bit 16
  TPB[23] TPB[22] TPB[21] TPB[20] TPB[19] TPB[18] TPB[17] TPB[16]

  Bit 15  Bit 14  Bit 13  Bit 12  Bit 11  Bit 10  Bit 9 Bit 8
  TPB[15] TPB[14] TPB[13] TPB[12] TPB[11] TPB[10] TPB[9]  TPB[8]

  Bit 7   Bit 6   Bit 5   Bit 4   Bit 3   Bit 2   Bit 1   Bit 0
  TPB[7]  TPB[6]  TPB[5]  TPB[4]  TPB[3]  TPB[2]  TPB[1]  TPB[0]

RAW = 1 This file contains raw (=not processed) data - used for hardware debugging only.
    = 0  processed data - only this file format is described here.

TPB[23:0] Time per bin.
    This is the time unit for this file. All times are expressed in multiples of this unit.
    TPB is expressed in femto seconds (1 fs = 10-15 s)


All events are recorded with their channel number and a 54 bit integer time stamp.
The time stamp reflects the absolute time from the start of the measurement.
The time unit is the TPB entry in the descriptor (see above).
To save disk space an event is stored with the lower 24 bit of the time stamp only.
To be able to reconstruct the complete time stamp another type of records
  is defined: TIME[53:24]. This record does not decribe events.
  It is a command to set the high part of the event time.
Both parts TIME[53:24] and TIME[23:0] (contained in the Event Record)
    together are the complete time stamp of an event.

-------------------------------------------------------------------------------
Record Types
-------------------------------------------------------------------------------
  Bit 31  Bit 30  Type         Description
     0       0    Event        Low Part of Time [TPB] incl. channel number
     0       1    TIME[53:24]  Set middle part of Time to TIME[53:24]
     1       0    reserved     reserved
     1       1    Special      Currently only Descriptor at start of file

-------------------------------------------------------------------------------
Event Record
-------------------------------------------------------------------------------
  Bit 31  Bit 30  Bit 29  Bit 28  Bit 27  Bit 26  Bit 25  Bit 24
     0       0      GAP    Ch[4]   CH[3]   CH[2]   CH[1]   CH[0]

  Bit 23    Bit 22    Bit 21    Bit 20    Bit 19    Bit 18    Bit 17    Bit 16
  TIME[23]  TIME[22]  TIME[21]  TIME[20]  TIME[19]  TIME[18]  TIME[17]  TIME[16]

  Bit 15    Bit 14    Bit 13    Bit 12    Bit 11    Bit 10    Bit 9     Bit 8
  TIME[15]  TIME[14]  TIME[13]  TIME[12]  TIME[11]  TIME[10]  TIME[9]   TIME[8]

  Bit 7     Bit 6     Bit 5     Bit 4     Bit 3     Bit 2     Bit 1     Bit 0
  TIME[7]   TIME[6]   TIME[5]   TIME[4]   TIME[3]   TIME[2]   TIME[1]   TIME[0]

  GAP         - Data lost before this entry due to full FIFO (1) / no gap occurred (0)
  CH[4:0]     - Channel Number ( 1-20)
  TIME[23:0]  - Event Time [TPB] - low part


-------------------------------------------------------------------------------
TIME[53:24] Record
-------------------------------------------------------------------------------
  Bit 31    Bit 30    Bit 29    Bit 28    Bit 27    Bit 26    Bit 25    Bit 24
     0         1      TIME[53]  TIME[52]  TIME[51]  TIME[50]  TIME[49]  TIME[48]

  Bit 23    Bit 22    Bit 21    Bit 20    Bit 19    Bit 18    Bit 17    Bit 16
  TIME[47]  TIME[46]  TIME[45]  TIME[44]  TIME[43]  TIME[42]  TIME[41]  TIME[40]

  Bit 15    Bit 14    Bit 13    Bit 12    Bit 11    Bit 10    Bit 9     Bit 8
  TIME[39]  TIME[38]  TIME[37]  TIME[36]  TIME[35]  TIME[34]  TIME[33]  TIME[32]

  Bit 7     Bit 6     Bit 5     Bit 4     Bit 3     Bit 2     Bit 1     Bit 0
  TIME[31]  TIME[30]  TIME[29]  TIME[28]  TIME[27]  TIME[26]  TIME[25]  TIME[24]

TIME[53:24] Set Event Time [TPB] - high part

*/




/////////////////////////////////////////////////////////////////////////
////////////         .ph   files structure       ////////////////////////
/////////////////////////////////////////////////////////////////////////

/*

Measurements in the FIFO mode can be done by user's own program using SPCM DLL
From SPCM DLL v.4.0 a new set of functions to control FIFO measurements is available
The functions use buffered photons stream to extract photons from the buffers
  during running FIFO measurement, without necessity to create .spc files
  (see use_spcm.c or use_dpc.c files for examples of such measurements).
Function SPC_get_photons_from_stream fills a buffer with photons data
   represented by PhotInfo64 structure.

typedef struct {
  unsigned __int64  mtime;   // macro time clocks 64 bits
  unsigned short micro_time; // micro time = 4095(255) - ADC value,  0-255 or 0-4095,
                             //   0 for markers entries, not valid for DPC-230
  unsigned short rout_chan;  // routing channel, 0-15,
                             //   (0 - 19 for DPC-230, 0-255 for SPC-6x0 48-bit mode ),
                             //    0 for markers entries
                             //    for QC modules bits 4-6 = Input Channel number (0 - 3(5) )
  unsigned short flags;      // photon flags, see defines in spcm_def.h file
  }PhotInfo64;

User can create .ph file by saving a buffer containing photons PhotInfo64 structures
   to a file with extension .ph. At the beginning of the file the header must be inserted
   ( the same header as for .spc file - see above for definition of first 4(6) bytes of .spc file).

The contents of the .ph file looks like below:
  - File header ( 4(6) bytes as in .spc file
  - PhotInfo64 structure of photon 0
  - PhotInfo64 structure of photon 1
  ..
  ..
  - PhotInfo64 structure of last photon

.ph files with the structure described above are accepted by 'Convert FIFO files' in SPCM software.
It should be used together with a proper .set file. It can be created using SPCM software
  in simulation mode of user's module type and setting FIFO mode.
By using 'Convert FIFO files' feature user can create different histograms for the photons collected
  by user's software ( which uses SPCM DLL) and create .sdt file with the histograms.

*/





//////////////////////////////////////////////////////////////
///    FIFO Data Files (SPC-QC-104, SPC-QC-004 )
//////////////////////////////////////////////////////////////

/*

--------------------------------------------------------
Data format in regular FIFO stream

The different FIFO events will be encoded in the following way:

Photons:
Bit      31  30   29     28    27 .. 16   15 .. 12   11 .. 0
function  0   0  Ch[1]  Ch[0]  Microtime  Routing    Macrotime

Macrotime overflow:
Bit      31  30   29    28     27 .. 16   15 .. 12   11 .. 0
function  1   0    0     0       0x0         0x0        0x0

Marker:
Bit      31  30   29    28     27 .. 16   15 .. 12   11 .. 0
function  0   1    0     0       0x0       Marker    Macrotime

GAP:
Bit      31  30   29    28     27 .. 16   15 .. 12   11 .. 0
function  1   1  Ch[1] Ch[0]   Microtime  Routing    Macrotime

It is important to know, that Macrotime overflows are never combined with photons
    and the GAP event is always the last Photon event before the FIFO-overflow.

Macrotime clock  = 2.048ns


--------------------------------------------------
Data format in absolute time mode

In the absolute time mode, the FIFO data format is largely maintained,
  the only thing different is the Microtime for photons and GAP events.
This does not depend on resolution or range anymore but just contains the missing lower bits of the time measurement,
  which are not covered in the macrotime.
In that way, the Macrotime contains bits [21 .. 9] and the Microtime bits [8 .. 0] of the absolute time measurement.

The resolution of this value (macrotime clock ) = 4ps

 The two values can just be stitched together as:
    T[s] = ( Macrotime * 512  + Microtime ) * 4e-12;

Photons/GAPs:
Bit      31  30   29    28          27 .. 16              15 .. 12   11 .. 0
function  0   0  Ch[1] Ch[0]  “000” – Microtime[8 .. 0]   Routing    Macrotime


The first photon frame (4 bytes) in the .spc file is added by the software and contains flags
  to recognize type of fifo data

    bit  31  = 1 - invalid flag
    bits 30-27 = no of rout bits
    bits 26  = 1 - raw data file  - QC .spc files are always raw ( not processed)
    bit  25      - file with markers (created in imaging mode)
    bit  24  = 1 - femto flag, femto units of MT clock
    bit  23      - 6 input channel module ( SPC-QC-106/006 )
    bit  22      - not used yet
    bits 21-0    - macro time clock in 1fs( femto, 1e-15 ) units


*/



//////////////////////////////////////////////////////////////
///    FIFO Data Files (SPC-QC-106, SPC-QC-006 )
//////////////////////////////////////////////////////////////

/*

--------------------------------------------------------
Data format in regular FIFO stream

The different FIFO events will be encoded in the following way:

Photons:
Bit      31    30     29     28     27 .. 16    15 .. 12     11 .. 0
function  0   Ch[2]  Ch[1]  Ch[0]   Microtime    Routing     Macrotime

Macrotime overflow:
Bit      31    30     29     28     27 .. 16    15 .. 12     11 .. 0
function  1     0      0      0       0x0          0x0          0x0

Marker:
Bit      31    30     29     28     27 .. 16    15 .. 12     11 .. 0
function  1     0      1      0        0x0       Marker      Macrotime

GAP:
Bit      31    30     29     28     27 .. 16    15 .. 12     11 .. 0
function  1     1    Ch[1]  Ch[0]   Microtime    Routing     Macrotime

It is important to know, that Macrotime overflows are never combined with photons
    and the GAP event is always the last Photon event before the FIFO-overflow.

Macrotime clock  = 2.048ns


--------------------------------------------------
Data format in absolute time mode

In the absolute time mode, the FIFO data format is largely maintained,
  the only thing different is the Microtime for photons and GAP events.
This does not depend on resolution or range anymore but just contains the missing lower bits of the time measurement,
  which are not covered in the macrotime.
In that way, the Macrotime contains bits [21 .. 9] and the Microtime bits [8 .. 0] of the absolute time measurement.

The resolution of this value (macrotime clock ) = 4ps

 The two values can just be stitched together as:
    T[s] = ( Macrotime * 512  + Microtime ) * 4e-12;

Photons/GAPs:
Bit      31    30     29    28          27 .. 16              15 .. 12    11 .. 0
function  0   Ch[2]  Ch[1] Ch[0]  “000” – Microtime[8 .. 0]   Routing     Macrotime

The first photon frame (4 bytes) in the .spc file is added by the software and contains flags
  to recognize type of fifo data

    bit  31  = 1 - invalid flag
    bits 30-27 = no of rout bits
    bits 26  = 1 - raw data file  - QC .spc files are always raw ( not processed)
    bit  25      - file with markers (created in imaging mode)
    bit  24  = 1 - femto flag, femto units of MT clock
    bit  23  = 1 - 6 input channel module ( SPC-QC-106/006 )
    bit  22      - not used yet
    bits 21-0    - macro time clock in 1fs( femto, 1e-15 ) units


*/



