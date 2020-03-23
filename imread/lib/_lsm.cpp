/*=========================================================================

  Program:   BioImageXD
  Language:  C++

 This is an open-source copyright as follows:
 Copyright (c) 2004-2008 BioImageXD Development Team
 Copyright (C) 2012-2020 Luis Pedro Coelho

 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Modified source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
 IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.


=========================================================================*/

#include "_lsm.h"
namespace {
#include "lzw.cpp"
}

#define TIF_NEWSUBFILETYPE 254
#define TIF_IMAGEWIDTH 256
#define TIF_IMAGELENGTH 257
#define TIF_BITSPERSAMPLE 258
#define TIF_COMPRESSION 259
#define TIF_PHOTOMETRICINTERPRETATION 262
#define TIF_STRIPOFFSETS 273
#define TIF_SAMPLESPERPIXEL 277
#define TIF_STRIPBYTECOUNTS 279
#define TIF_PLANARCONFIGURATION 284
#define TIF_PREDICTOR 317
#define TIF_COLORMAP 320
#define TIF_CZ_LSMINFO 34412

#define SUBBLOCK_END        0x0FFFFFFFF
#define SUBBLOCK_RECORDING  0x010000000
#define SUBBLOCK_LASERS     0x030000000
#define SUBBLOCK_LASER      0x050000000
#define SUBBLOCK_TRACKS     0x020000000
#define SUBBLOCK_TRACK      0x040000000
#define SUBBLOCK_DETECTION_CHANNELS      0x060000000
#define SUBBLOCK_DETECTION_CHANNEL       0x070000000
#define SUBBLOCK_ILLUMINATION_CHANNELS   0x080000000
#define SUBBLOCK_ILLUMINATION_CHANNEL    0x090000000
#define SUBBLOCK_BEAM_SPLITTERS          0x0A0000000
#define SUBBLOCK_BEAM_SPLITTER           0x0B0000000
#define SUBBLOCK_DATA_CHANNELS           0x0C0000000
#define SUBBLOCK_DATA_CHANNEL            0x0D0000000
#define SUBBLOCK_TIMERS                  0x011000000
#define SUBBLOCK_TIMER                   0x012000000
#define SUBBLOCK_MARKERS                 0x013000000
#define SUBBLOCK_MARKER                  0x014000000

#define RECORDING_ENTRY_NAME            0x010000001
#define RECORDING_ENTRY_DESCRIPTION     0x010000002
#define RECORDING_ENTRY_NOTES           0x010000003
#define RECORDING_ENTRY_OBJETIVE        0x010000004
#define RECORDING_ENTRY_PROCESSING_SUMMARY  0x010000005
#define RECORDING_ENTRY_SPECIAL_SCAN_MODE   0x010000006
#define RECORDING_ENTRY_SCAN_TYPE           0x010000007
#define OLEDB_RECORDING_ENTRY_SCAN_MODE     0x010000008
#define RECORDING_ENTRY_NUMBER_OF_STACKS    0x010000009
#define RECORDING_ENTRY_LINES_PER_PLANE     0x01000000A
#define RECORDING_ENTRY_SAMPLES_PER_LINE    0x01000000B
#define RECORDING_ENTRY_PLANES_PER_VOLUME   0x01000000C
#define RECORDING_ENTRY_IMAGES_WIDTH        0x01000000D
#define RECORDING_ENTRY_IMAGES_HEIGHT       0x01000000E
#define RECORDING_ENTRY_IMAGES_NUMBER_PLANES 0x01000000F
#define RECORDING_ENTRY_IMAGES_NUMBER_STACKS 0x010000010
#define RECORDING_ENTRY_IMAGES_NUMBER_CHANNELS 0x010000011
#define RECORDING_ENTRY_LINSCAN_XY_SIZE     0x010000012
#define RECORDING_ENTRY_SCAN_DIRECTION      0x010000013
#define RECORDING_ENTRY_TIME_SERIES         0x010000014
#define RECORDING_ENTRY_ORIGINAL_SCAN_DATA  0x010000015
#define RECORDING_ENTRY_ZOOM_X              0x010000016
#define RECORDING_ENTRY_ZOOM_Y              0x010000017
#define RECORDING_ENTRY_ZOOM_Z              0x010000018
#define RECORDING_ENTRY_SAMPLE_0X           0x010000019
#define RECORDING_ENTRY_SAMPLE_0Y           0x01000001A
#define RECORDING_ENTRY_SAMPLE_0Z           0x01000001B
#define RECORDING_ENTRY_SAMPLE_SPACING      0x01000001C
#define RECORDING_ENTRY_LINE_SPACING        0x01000001D
#define RECORDING_ENTRY_PLANE_SPACING       0x01000001E
#define RECORDING_ENTRY_PLANE_WIDTH         0x01000001F
#define RECORDING_ENTRY_PLANE_HEIGHT        0x010000020
#define RECORDING_ENTRY_VOLUME_DEPTH        0x010000021
#define RECORDING_ENTRY_ROTATION            0x010000034
#define RECORDING_ENTRY_NUTATION            0x010000023
#define RECORDING_ENTRY_PRECESSION          0x010000035
#define RECORDING_ENTRY_SAMPLE_0TIME        0x010000036


#define LASER_ENTRY_NAME                         0x050000001
#define LASER_ENTRY_ACQUIRE                      0x050000002
#define LASER_ENTRY_POWER                        0x050000003

#define DETCHANNEL_ENTRY_DETECTOR_GAIN_FIRST     0x070000003
#define DETCHANNEL_ENTRY_DETECTOR_GAIN_LAST      0x070000004
#define DETCHANNEL_ENTRY_INTEGRATION_MODE        0x070000001
#define DETCHANNEL_ENTRY_ACQUIRE                 0x07000000B
#define DETCHANNEL_DETECTION_CHANNEL_NAME        0x070000014

#define ILLUMCHANNEL_ENTRY_WAVELENGTH            0x090000003
#define ILLUMCHANNEL_ENTRY_AQUIRE                0x090000004
#define ILLUMCHANNEL_DETCHANNEL_NAME             0x090000005

#define TRACK_ENTRY_ACQUIRE                      0x040000006
#define TRACK_ENTRY_NAME                         0x04000000C
#define TYPE_SUBBLOCK   0
#define TYPE_LONG       4
#define TYPE_RATIONAL   5
#define TYPE_ASCII      2


#define TIFF_BYTE 1
#define TIFF_ASCII 2
#define TIFF_SHORT 3
#define TIFF_LONG 4
#define TIFF_RATIONAL 5

#define LSM_MAGIC_NUMBER 42

#define LSM_COMPRESSED 5

namespace {

class LSMReader {
    public:
        LSMReader(byte_source* s);
        ~LSMReader();

        void PrintSelf(std::ostream& os, const char* indent="");
        std::unique_ptr<Image> read(ImageFactory* factory, const options_map&);
        void readHeader();

        int GetChannelColorComponent(int,int);
        std::string GetChannelName(int);

        void SetDataByteOrderToBigEndian();
        void SetDataByteOrderToLittleEndian();

        int GetDataTypeForChannel(unsigned int channel);

    private:

        static int FindChannelNameStart(const byte *, int);
        static int ReadChannelName(const byte *, int, byte *);

        unsigned long ReadImageDirectory(byte_source *,unsigned long);
        int ReadChannelDataTypes(byte_source*, unsigned long);
        int ReadChannelColorsAndNames(byte_source *,unsigned long);
        int ReadTimeStampInformation(byte_source *,unsigned long);
        int ReadLSMSpecificInfo(byte_source *,unsigned long);
        int AnalyzeTag(byte_source *,unsigned long);
        int ReadScanInformation(byte_source*, unsigned long);
        unsigned long GetOffsetToImage(int, int);


        void CalculateExtentAndSpacing(int extent[6],double spacing[3]);
        void DecodeHorizontalDifferencing(unsigned char *,int);
        void DecodeHorizontalDifferencingUnsignedShort(unsigned short*, int);
        void DecodeLZWCompression(unsigned  char *,int, int);
        void ConstructSliceOffsets(int channel);
        unsigned int GetStripByteCount(unsigned int timepoint, unsigned int slice);
        unsigned int GetSliceOffset(unsigned int timepoint, unsigned int slice);


        byte_source* src;
        bool swap_bytes_;

        unsigned long OffsetToLastAccessedImage;
        int NumberOfLastAccessedImage;
        double VoxelSizes[3];
        int dimensions_[5];// x,y,z,time,channels
        int NumberOfIntensityValues[4];
        unsigned int NewSubFileType;
        std::vector<unsigned short> bits_per_sample_;
        unsigned int compression_;
        std::vector<unsigned int> strip_offset_;
        std::vector<unsigned int> channel_data_types_;
        std::vector<double> track_wavelengths_;
        unsigned int sample_per_pixel_;
        std::vector<unsigned int> strip_byte_count_;
        unsigned int LSMSpecificInfoOffset;
        unsigned short PhotometricInterpretation;
        unsigned long ColorMapOffset;
        unsigned short PlanarConfiguration;
        unsigned short Predictor;
        unsigned short scan_type_;

        std::vector<unsigned int> image_offsets_;
        std::vector<unsigned int> read_sizes_;
        std::vector<double> detector_offset_first_image_;
        std::vector<double> detector_offset_last_image_;
        std::vector<std::string> laser_names_;

        double DataSpacing[3];
        int DataExtent[6];
        int data_type_;
        unsigned long ChannelInfoOffset;
        unsigned long ChannelDataTypesOffset;
        std::vector<int> channel_colors_;
        std::vector<std::string> channel_names_;
        std::vector<double> time_stamp_info_;
        std::string objective_;
        std::string description_;
        double TimeInterval;

};

int ReadFile(byte_source* s, unsigned long *pos, int size, void *buf, bool swap=false)
{
    s->seek_absolute(*pos);
    const unsigned ret = s->read(reinterpret_cast<byte*>(buf), size);
#ifdef VTK_WORDS_BIGENDIAN
  if(swap) {
    vtkByteSwap::SwapLERange(buf,size);
  }
#endif
    *pos += ret;
    return ret;
}


std::string read_str(byte_source* s, unsigned long* pos, const unsigned long len) {
    char* buf = new char[len];
    ReadFile(s, pos, len, buf, 1);
    std::string res(buf, len);
    delete [] buf;
    return res;
}


int CharPointerToInt(char *buf)
{
  return *((int*)(buf));
}


uint32_t parse_uint32_t(const byte *buf) {
#ifdef VTK_WORDS_BIGENDIAN
    char buf2[4];
    ::memcpy(buf2, buf, 4);
    vtkByteSwap::Swap4LE(buf2);
    return *reinterpret_cast<const uint32_t*>(buf2);
#else
    return *reinterpret_cast<const uint32_t*>(buf);
#endif
}

uint32_t parse_uint32_t(const std::vector<byte> &buf) {
    if (buf.size() < 4) {
        throw CannotReadError("Malformed LSM file: expected 4 Bytes, cannot parse uint32_t");
    }
    return parse_uint32_t(buf.data());
}

short CharPointerToShort(char *buf)
{
  return *((short*)(buf));
}


uint16_t parse_uint16_t(const byte *buf) {
#ifdef VTK_WORDS_BIGENDIAN
    char buf2[2];
    buf2[0] = buf[1];
    buf2[1] = buf[0];
    return *reinterpret_cast<const uint16_t*>(buf2);
#else
    return *reinterpret_cast<const uint16_t*>(buf);
#endif
}
uint16_t parse_uint16_t(const std::vector<byte>& buf) {
    if (buf.size() < 2) {
        throw CannotReadError("Failed to read short (size(vec) < 2)");
    }
    return parse_uint16_t(buf.data());
}

double CharPointerToDouble(char *buf)
{
  return *((double*)(buf));
}

int ReadInt(byte_source* s, unsigned long *pos)
{
  char buff[4];
  ReadFile(s,pos,4,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap4LE((int*)buff);
#endif
  return CharPointerToInt(buff);
}

unsigned int ReadUnsignedInt(byte_source* s, unsigned long *pos) {
    byte buff[4];
    ReadFile(s, pos, 4, buff);
    return parse_uint32_t(buff);
}

short ReadShort(byte_source* s, unsigned long *pos)
{
  char buff[2];
  ReadFile(s,pos,2,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap2LE((short*)buff);
#endif
  return CharPointerToShort(buff);
}

uint16_t ReadUnsignedShort(byte_source* s, unsigned long *pos) {
    byte buff[2];
    ReadFile(s,pos,2,buff);
    return parse_uint16_t(buff);
}

double ReadDouble(byte_source* s, unsigned long *pos)
{
  char buff[8];
  ReadFile(s,pos,8,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap8LE((double*)buff);
#endif
  return CharPointerToDouble(buff);
}


int BYTES_BY_DATA_TYPE(int type) {
    switch(type) {
        case 1: return 1;
        case 2: return 2;
        case 3: return 2;
        case 5: return 4;
    }
    return 1;
}
int TIFF_BYTES(unsigned short type)
{
    switch (type) {
        case TIFF_BYTE:
            return 1;
        case TIFF_ASCII:
        case TIFF_SHORT:
            return 2;
        case TIFF_LONG:
        case TIFF_RATIONAL:
            return 4;
    }
    return 1;
}


LSMReader::LSMReader(byte_source* s)
    :src(s)
    ,swap_bytes_(false)
    ,compression_(0)
    ,sample_per_pixel_(0)
    ,scan_type_(0)
    ,data_type_(0)
{
    std::fill(this->dimensions_, this->dimensions_ + 5, 0);

  this->DataExtent[0] = this->DataExtent[1] = this->DataExtent[2] = this->DataExtent[4] = 0;
  this->DataExtent[3] = this->DataExtent[5] = 0;
  this->OffsetToLastAccessedImage = 0;
  this->NumberOfLastAccessedImage = 0;
  this->VoxelSizes[0] = this->VoxelSizes[1] = this->VoxelSizes[2] = 0.0;

  this->DataSpacing[0] = this->DataSpacing[1] = this->DataSpacing[2] =  1.0f;


  this->NewSubFileType = 0;
  this->bits_per_sample_.resize(4);
  this->strip_offset_.resize(4);
  this->strip_byte_count_.resize(4);
  this->Predictor = 0;
  this->PhotometricInterpretation = 0;
  this->PlanarConfiguration = 0;
  this->ColorMapOffset = 0;
  this->LSMSpecificInfoOffset = 0;
  this->NumberOfIntensityValues[0] = this->NumberOfIntensityValues[1] = this->NumberOfIntensityValues[2] = this->NumberOfIntensityValues[3] = 0;
}

LSMReader::~LSMReader()
{
  this->channel_names_.clear();
  this->channel_colors_.clear();
  this->bits_per_sample_.clear();
  this->strip_offset_.clear();
  this->strip_byte_count_.clear();
  this->laser_names_.clear();
  this->track_wavelengths_.clear();
  this->channel_data_types_.clear();
  this->image_offsets_.clear();
  this->read_sizes_.clear();
}


void LSMReader::SetDataByteOrderToBigEndian()
{
#ifndef VTK_WORDS_BIGENDIAN
  this->swap_bytes_ = false;
#else
  this->swap_bytes_ = true;
#endif
}

void LSMReader::SetDataByteOrderToLittleEndian()
{
#ifdef VTK_WORDS_BIGENDIAN
  this->swap_bytes_ = true;
#else
  this->swap_bytes_ = false;
#endif
}



std::string LSMReader::GetChannelName(int chNum)
{
    if (chNum < 0 || unsigned(chNum) >= this->channel_names_.size()) return "";
    return this->channel_names_[chNum];
}


int LSMReader::FindChannelNameStart(const byte *buf, const int length) {
    for (int i = 0; i < length; ++i) {
        if (buf[i] > 32) return i;
    }
    return length;
}

int LSMReader::ReadChannelName(const byte *nameBuff, const int length, byte *buffer) {
    for (int i = 0; i < length; ++i) {
        buffer[i] = nameBuff[i];
        if (!buffer[i]) return i;
    }
    return length;
}

int LSMReader::ReadChannelDataTypes(byte_source* s, unsigned long start)
{
    const unsigned int numOfChls = this->dimensions_[4];
    this->channel_data_types_.resize(numOfChls);

    unsigned long pos = start;
    for(unsigned int i=0; i < numOfChls; i++) {
        this->channel_data_types_[i] = ReadUnsignedInt(s, &pos);
    }
    return 0;
}

int LSMReader::ReadChannelColorsAndNames(byte_source* s, const unsigned long start) {
    unsigned long pos = start;
    // Read size of structure
    const int sizeOfStructure = ReadInt(s, &pos);
    // Read number of colors
    const int n_cols = ReadInt(s, &pos);
    // Read number of names
    const int n_names = ReadInt(s, &pos);
    const int sizeOfNames = sizeOfStructure - ( (10*4) + (n_cols*4) );

    if(n_cols != this->dimensions_[4]) {
        throw CannotReadError("LSM file seems malformed (n_cols != dimensions_[4])");
    }
    if(n_names != this->dimensions_[4]) {
        throw CannotReadError("LSM file seems malformed (n_names != dimensions_[4])");
    }

    // Read offset to color info
    unsigned long colorOffset = ReadInt(s, &pos) + start;
    // Read offset to name info
    unsigned long nameOffset = ReadInt(s, &pos) + start;

    this->channel_colors_.resize( 3 *  (n_cols+1));

  // Read the colors
    for(int j = 0; j < this->dimensions_[4]; ++j) {
        char colorBuff[5];
        ReadFile(s, &colorOffset, 4, colorBuff, 1);
        for(int i=0;i<3;i++) {
            this->channel_colors_[i + 3*j] = static_cast<unsigned char>(colorBuff[i]);
        }
    }


    std::vector<byte> nameBuff;
    nameBuff.resize(sizeOfNames + 1);
    std::vector<byte> name;
    name.resize(sizeOfNames + 1);

    ReadFile(s, &nameOffset, sizeOfNames, nameBuff.data(), 1);

    this->channel_names_.resize(this->dimensions_[4]);
    int nameStart = 0;
    for(int i = 0; i < this->dimensions_[4]; i++) {
        nameStart += FindChannelNameStart(nameBuff.data() + nameStart,
                                            sizeOfNames-nameStart);
        if (nameStart >= sizeOfNames) {
            throw CannotReadError("LSM file malformed");
        }
        const int nameLength = ReadChannelName(nameBuff.data()+nameStart, sizeOfNames-nameStart, name.data());
        nameStart += nameLength;

        this->channel_names_[i] = std::string(reinterpret_cast<const char*>(name.data()));
    }
    return 0;
}

int LSMReader::ReadTimeStampInformation(byte_source* s, unsigned long offset) {
    // position is 0 for non-timeseries files!
    if( offset == 0 ) return 0;

    offset += 4;
    int numOffStamps = ReadInt(s,&offset);

    this->time_stamp_info_.resize(numOffStamps);
    for(int i=0;i<numOffStamps;i++)
    {
        this->time_stamp_info_[i] = ReadDouble(s,&offset);
    }
  return 0;
}

/* Read the TIF_CZ_LSMINFO entry described in Table 17 of the LSM file format specification
 *
 *
 */
int LSMReader::ReadLSMSpecificInfo(byte_source* s, unsigned long pos) {

  pos += 2 * 4; // skip over the start of the LSMInfo
                // first 4 byte entry if magic number
                // second is number of bytes in this structure

  // Then we read X
  this->NumberOfIntensityValues[0] = ReadInt(s,&pos);

  // vtkByteSwap::Swap4LE((int*)&this->NumberOfIntensityValues[0]);
  this->dimensions_[0] = this->NumberOfIntensityValues[0];
  // Y
  this->NumberOfIntensityValues[1] = ReadInt(s,&pos);
  this->dimensions_[1] = this->NumberOfIntensityValues[1];
  // and Z dimension
  this->NumberOfIntensityValues[2] = ReadInt(s,&pos);
  this->dimensions_[2] = this->NumberOfIntensityValues[2];
  // Read number of channels
  this->dimensions_[4] = ReadInt(s,&pos);

  // Read number of timepoints
  this->NumberOfIntensityValues[3] = ReadInt(s,&pos);
  this->dimensions_[3] = this->NumberOfIntensityValues[3];

  // Read datatype, 1 for 8-bit unsigned int
  //                2 for 12-bit unsigned int
  //                5 for 32-bit float (timeseries mean of ROIs)
  //                0 if the channels have different types
  //                In that case, u32OffsetChannelDataTypes
  //                has further info
  this->data_type_ = ReadInt(s,&pos);

  // Skip the width and height of thumbnails
  pos += 2 * 4;

  // Read voxel sizes
  this->VoxelSizes[0] = ReadDouble(s,&pos);
  this->VoxelSizes[1] = ReadDouble(s,&pos);
  this->VoxelSizes[2] = ReadDouble(s,&pos);

  // Skip over OriginX,OriginY,OriginZ which are not used
  pos += 3*8;

  // Read scan type which is
  // 0 for normal x-y-z scan
  // 1 for z-scan (x-z plane)
  // 2 for line scan
  // 3 for time series x-y
  // 4 for time series x-z
  // 5 time series mean of ROIs
  // 6 time series x y z
  // 7 spline scan
  // 8 spline plane x-z
  // 9 time series spline plane
  // 10 point mode
  this->scan_type_ = ReadShort(s,&pos);

  if (this->scan_type_ == 1)
	{
	  int tmp = this->dimensions_[1];
	  this->dimensions_[1] = this->dimensions_[2];
	  this->dimensions_[2] = tmp;
	}

  // skip over SpectralScan flag
  // if 0, no spectral scan
  // if 1, image has been acquired with spectral scan mode with a "meta" detector
  // skip over DataType, Offset to vector overlay, Offset to input LUT
  pos += 1*2 + 4*4;// + 1*8 + 3*4;

  // Read OffsetChannelColors, which is an offset to channel colors and names
  this->ChannelInfoOffset = ReadUnsignedInt(s,&pos);
  if (this->ChannelInfoOffset != 0)
	this->ReadChannelColorsAndNames(s,this->ChannelInfoOffset);

  // Skip time interval in seconds (8 bytes)
  //pos += 1*8;
  this->TimeInterval = ReadDouble(s, &pos);

  // If each channel has different datatype (meaning DataType == 0), then
  // read the offset to more information and read the info
  this->ChannelDataTypesOffset = ReadInt(s, &pos);
  unsigned long scanInformationOffset = ReadUnsignedInt(s, &pos);
  if(this->data_type_ == 0) {
    this->ReadChannelDataTypes(s, this->ChannelDataTypesOffset);
  }

  // Read scan information
  this->ReadScanInformation(s, scanInformationOffset);
  // SKip Zeiss Vision KS-3D speific data
  pos +=  4;
  // Read timestamp information
  unsigned long offset = ReadUnsignedInt(s, &pos);
  this->ReadTimeStampInformation(s,offset);

  return 1;
}
int LSMReader::ReadScanInformation(byte_source* s,  unsigned long pos)
{
    unsigned int subblocksOpen = 0;
    double wavelength;
    int isOn = 0;
    do {
        const unsigned int entry = ReadUnsignedInt(s, &pos);
        const unsigned int type =  ReadUnsignedInt(s, &pos);
        const unsigned int size =  ReadUnsignedInt(s, &pos);

        if (type == TYPE_SUBBLOCK) {
            if (entry == SUBBLOCK_END) --subblocksOpen;
            else ++subblocksOpen;
        }

        switch(entry) {
            case DETCHANNEL_ENTRY_DETECTOR_GAIN_FIRST:
                (void)ReadDouble(s, &pos);
                continue;
                break;
            case DETCHANNEL_ENTRY_DETECTOR_GAIN_LAST:
                (void)ReadDouble(s, &pos);
                continue;
                break;
            case DETCHANNEL_ENTRY_INTEGRATION_MODE:
                (void)ReadInt(s, &pos);
                continue;
                break;
            case LASER_ENTRY_NAME:
                this->laser_names_.push_back(
                            read_str(s, &pos, size)
                            );
                continue;
                break;
            case ILLUMCHANNEL_ENTRY_WAVELENGTH:
                wavelength = ReadDouble(s, &pos);

                continue;
                break;
            case ILLUMCHANNEL_DETCHANNEL_NAME:
                (void)read_str(s, &pos, size);
                continue;
                break;
            case TRACK_ENTRY_ACQUIRE:
                (void)ReadInt(s, &pos);

                continue;
                break;
            case TRACK_ENTRY_NAME:
                (void)read_str(s, &pos, size);
                continue;
                break;
            case DETCHANNEL_DETECTION_CHANNEL_NAME:
                (void)read_str(s, &pos, size);
                continue;
                break;
            case DETCHANNEL_ENTRY_ACQUIRE:
                (void)ReadInt(s, &pos);
                continue;
                break;

            case ILLUMCHANNEL_ENTRY_AQUIRE:
                isOn = ReadInt(s, &pos);
                if(isOn) {
                     this->track_wavelengths_.push_back(wavelength);
                }
                continue;
                break;
            case RECORDING_ENTRY_DESCRIPTION:
                this->description_ = read_str(s, &pos, size);
                continue;
                break;
            case RECORDING_ENTRY_OBJETIVE:
                this->objective_ = read_str(s, &pos, size);
                continue;

            case SUBBLOCK_RECORDING:
                break;
            case SUBBLOCK_LASERS:
                break;
            case SUBBLOCK_LASER:
                break;
            case SUBBLOCK_TRACKS:
                break;
            case SUBBLOCK_TRACK:
                break;
            case SUBBLOCK_DETECTION_CHANNELS:
                break;
            case SUBBLOCK_DETECTION_CHANNEL:
                break;
            case SUBBLOCK_ILLUMINATION_CHANNELS:
                break;
            case SUBBLOCK_ILLUMINATION_CHANNEL:
                break;
            case SUBBLOCK_BEAM_SPLITTERS:
                break;
            case SUBBLOCK_BEAM_SPLITTER:
                break;
            case SUBBLOCK_DATA_CHANNELS:
                break;
            case SUBBLOCK_DATA_CHANNEL:
                break;
            case SUBBLOCK_TIMERS:
                break;
            case SUBBLOCK_TIMER:
                break;
            case SUBBLOCK_MARKERS:
                break;
            case SUBBLOCK_MARKER:
                break;
        }
        pos += size;
    } while (subblocksOpen);
    return 0;
}

int LSMReader::AnalyzeTag(byte_source* s, unsigned long startPos) {
    std::vector<byte> valueData;
    const unsigned short tag = ReadUnsignedShort(s, &startPos);
    const unsigned short type = ReadUnsignedShort(s, &startPos);
    const unsigned short length = ReadUnsignedInt(s, &startPos);

    valueData.resize(4);
    ReadFile(s, &startPos, 4, valueData.data());

    const int value = parse_uint32_t(valueData);

    // if there is more than 4 bytes in value,
    // value is an offset to the actual data
    const int dataSize = TIFF_BYTES(type);
    const unsigned long readSize = dataSize*length;
    if(readSize > 4 && tag != TIF_CZ_LSMINFO) {
        valueData.resize(readSize);
        startPos = value;
        if(tag == TIF_STRIPOFFSETS ||tag == TIF_STRIPBYTECOUNTS) {
            if( !ReadFile(s, &startPos, readSize, valueData.data()) ) {
                throw CannotReadError("Failed to get strip offsets\n");
            }
        }
    }
    switch(tag) {
        case TIF_NEWSUBFILETYPE:
            this->NewSubFileType = value;
            break;

        case TIF_IMAGEWIDTH:
            //this->dimensions_[0] = parse_uint32_t(valueData);
            //this->dimensions_[0] = value;
            break;

        case TIF_IMAGELENGTH:
            //this->dimensions_[1] = parse_uint32_t(valueData);
            //this->dimensions_[1] = value;
            break;

        case TIF_BITSPERSAMPLE:
            if (valueData.size() < TIFF_BYTES(TIFF_SHORT) * length) {
                throw CannotReadError("LSM file is malformed (TIF_BITSPERSAMPLE field is too short)");
            }
            this->bits_per_sample_.resize(length);
            for(int i=0;i<length;i++) {
                this->bits_per_sample_[i] = parse_uint16_t(valueData.data() + TIFF_BYTES(TIFF_SHORT)*i);
            }
            break;

        case TIF_COMPRESSION:
            this->compression_ = parse_uint16_t(valueData);
            break;

        case TIF_PHOTOMETRICINTERPRETATION:
            this->PhotometricInterpretation = parse_uint16_t(valueData);
            break;

        case TIF_STRIPOFFSETS:
            this->strip_offset_.resize(length);
            if(length>1) {
                if (length * sizeof(uint32_t) > valueData.size()) {
                    throw CannotReadError("LSM file is malformed (TIF_STRIPOFFSETS field is too short)");
                }
                for(int i=0;i<length;++i) {
                    this->strip_offset_[i] = parse_uint32_t(valueData.data() + sizeof(uint32_t) * i);
                }
            } else {
                this->strip_offset_[0] = value;
            }
            break;

        case TIF_SAMPLESPERPIXEL:
            this->sample_per_pixel_ = parse_uint32_t(valueData);
            break;

        case TIF_STRIPBYTECOUNTS:
            this->strip_byte_count_.resize(length);
            if (length > 1) {
                for(int i=0; i<length; ++i) {
                    if (valueData.size() < TIFF_BYTES(TIFF_LONG) * i + 4) {
                        throw CannotReadError();
                    }
                    this->strip_byte_count_[i] = parse_uint32_t(valueData.data() + TIFF_BYTES(TIFF_LONG)*i);
                }
            } else {
                this->strip_byte_count_[0] = value;
            }
            break;
        case TIF_PLANARCONFIGURATION:
            this->PlanarConfiguration = parse_uint16_t(valueData);
            break;
        case TIF_PREDICTOR:
            this->Predictor = parse_uint16_t(valueData);
            break;
        case TIF_COLORMAP:
            //this->ColorMapOffset = parse_uint32_t(valueData);
            break;
        case TIF_CZ_LSMINFO:

            this->LSMSpecificInfoOffset = value;
            break;
    }

    return 0;
}



unsigned int LSMReader::GetStripByteCount(unsigned int timepoint, unsigned int slice) {
    return this->read_sizes_[timepoint * this->dimensions_[2] + slice];
}

unsigned int LSMReader::GetSliceOffset(unsigned int timepoint, unsigned int slice) {
    return this->image_offsets_[timepoint * this->dimensions_[2] + slice];
}

void LSMReader::ConstructSliceOffsets(int channel) {
    this->image_offsets_.resize(this->dimensions_[2] * this->dimensions_[3]);
    this->read_sizes_.resize(this->dimensions_[2] * this->dimensions_[3]);

    for(int tp = 0; tp < this->dimensions_[3]; tp++) {
        for(int slice = 0; slice < this->dimensions_[2]; slice++) {
            this->GetOffsetToImage(slice, tp);
            this->image_offsets_[tp * this->dimensions_[2] + slice] =  this->strip_offset_[channel];
            this->read_sizes_[tp * this->dimensions_[2] + slice] = this->strip_byte_count_[channel];
        }
    }
}

unsigned long LSMReader::GetOffsetToImage(int slice, int timepoint)
{
  const int image = slice+(timepoint*this->dimensions_[2]);
  unsigned long offset = 4, finalOffset;
  int i=0;
  int imageCount = image+1;

  if(this->OffsetToLastAccessedImage && (this->NumberOfLastAccessedImage < image))
  {
    offset = this->OffsetToLastAccessedImage;
    imageCount = image - this->NumberOfLastAccessedImage;
  }
  else
  {
    offset = (unsigned long)ReadInt(this->src,&offset);
  }

  offset = this->ReadImageDirectory(this->src, offset);
  do
  {
    // we count only image directories and not thumbnail images
    // subfiletype 0 = images
    // subfiletype 1 = thumbnails
    if(this->NewSubFileType == 0)
    {
      i++;
    }
    finalOffset = offset;
    offset = this->ReadImageDirectory(this->src, offset);
  } while(i<imageCount && offset != 0);

  this->OffsetToLastAccessedImage = finalOffset;
  this->NumberOfLastAccessedImage = image;

  return finalOffset;
}

unsigned long LSMReader::ReadImageDirectory(byte_source* s, unsigned long offset)
{
  unsigned short numberOfTags=0;
  unsigned long nextOffset = offset;

  numberOfTags = ReadUnsignedShort(s,&offset);
  for(int i = 0; i < numberOfTags; i++)
  {
    this->AnalyzeTag(s,offset);
    if(this->NewSubFileType == 1) {
      break; //thumbnail image
    }
    offset = offset + 12;
  }
  nextOffset += 2 + numberOfTags * 12;
  return ReadUnsignedInt(s,&nextOffset);
}


void LSMReader::DecodeHorizontalDifferencing(unsigned char *buffer, int size)
{
  for(int i=1;i<size;i++)
    {
      *(buffer+i) = *(buffer+i) + *(buffer+i-1);
    }
}

void LSMReader::DecodeHorizontalDifferencingUnsignedShort(unsigned short *buffer, int size)
{
  for(int i=1;i<size;i++)
    {
      *(buffer+i) = *(buffer+i) + *(buffer+i-1);
    }
}

void LSMReader::DecodeLZWCompression(unsigned char* buffer, int size, int bytes) {
    throw ProgrammingError("Not tested");
    std::vector<unsigned char> decoded = lzw_decode(buffer, size);
    unsigned char* outbufp = &decoded[0];

    const int width = this->dimensions_[0];
    const int lines = size / (width*bytes);

    for(int line = 0; line < lines; line++) {
        if(this->Predictor == 2) {
            if(bytes == 1)
                this->DecodeHorizontalDifferencing(outbufp,width*bytes);
            else {
                this->DecodeHorizontalDifferencingUnsignedShort((unsigned short*)outbufp, width);
            }
        }
        outbufp += width*bytes;
    }
    for(int i=0;i < size;i++) {
        buffer[i] = decoded[i];
    }

}

int LSMReader::GetDataTypeForChannel(unsigned int channel)
{
    if (this->data_type_) return this->data_type_;
    if (this->channel_data_types_.empty()) return 1;
    return this->channel_data_types_.at(channel);
}

void LSMReader::readHeader() {
  this->SetDataByteOrderToLittleEndian();

    unsigned long startPos = 2;  // header identifier

    const unsigned short identifier = ReadUnsignedShort(this->src, &startPos);
    if (identifier != LSM_MAGIC_NUMBER) {
        throw CannotReadError("Given file is not a valid LSM-file (magic number mismatch).");
    }

    const unsigned int imageDirOffset = ReadUnsignedInt(this->src, &startPos);

    this->ReadImageDirectory(this->src, imageDirOffset);

    if (this->LSMSpecificInfoOffset) {
        ReadLSMSpecificInfo(this->src, (unsigned long)this->LSMSpecificInfoOffset);
    } else {
        throw CannotReadError("Did not found LSM specific info!");
    }
    if (!(this->scan_type_ == 6 || this->scan_type_ == 0 || this->scan_type_ == 3 || this->scan_type_ == 1) ) {
       throw CannotReadError("Sorry! Your LSM-file must be of type 6 LSM-file (time series x-y-z) "
                            "or type 0 (normal x-y-z) or type 3 (2D + time) or type 1 "
                            "(x-z scan). Type of this File is " /* % this->scan_type_ */);
    }

    this->CalculateExtentAndSpacing(this->DataExtent,this->DataSpacing);
}

void LSMReader::CalculateExtentAndSpacing(int extent[6],double spacing[3])
{
  extent[0] = extent[2] = extent[4] = 0;
  extent[1] = this->dimensions_[0] - 1;
  extent[3] = this->dimensions_[1] - 1;
  extent[5] = this->dimensions_[2] - 1;

  spacing[0] = int(this->VoxelSizes[0]*1000000);
  if (spacing[0] < 1.0) spacing[0] = 1.0;
  spacing[1] = this->VoxelSizes[1] / this->VoxelSizes[0];
  spacing[2] = this->VoxelSizes[2] / this->VoxelSizes[0];
}

//----------------------------------------------------------------------------

int LSMReader::GetChannelColorComponent(int ch, int component)
{
    if (ch < 0 ||
        component < 0 ||
        component > 2 ||
        unsigned(ch) > unsigned(this->dimensions_[4]-1) ||
        unsigned(ch) >= this->channel_colors_.size()) return 0;
  return this->channel_colors_[(ch*3) + component];
}

void LSMReader::PrintSelf(std::ostream& os, const char* indent)
{
  os << indent << "dimensions_: " << this->dimensions_[0] << "," << this->dimensions_[1] << ","<<this->dimensions_[2] << "\n";
  os << indent << "Time points: " << this->dimensions_[3] << "\n";
  os << "Number of channels: " << this->dimensions_[4] << "\n";
  os << "\n";
  os << indent << "Number of intensity values X: " << this->NumberOfIntensityValues[0] << "\n";
  os << indent << "Number of intensity values Y: " << this->NumberOfIntensityValues[1] << "\n";
  os << indent << "Number of intensity values Z: " << this->NumberOfIntensityValues[2] << "\n";
  os << indent << "Number of intensity values Time: " << this->NumberOfIntensityValues[3] << "\n";
  os << indent << "Voxel size X: " << this->VoxelSizes[0] << "\n";
  os << indent << "Voxel size Y: " << this->VoxelSizes[1] << "\n";
  os << indent << "Voxel size Z: " << this->VoxelSizes[2] << "\n";
  os << "\n";
  os << indent << "Scan type: " << this->scan_type_ << "\n";
  os << indent << "Data type: " << this->data_type_ << "\n";
  if(this->data_type_ == 0) {
     for(int i=0; i < this->dimensions_[4]; i++) {
        os << indent << indent << "Data type of channel "<<i<<": "<< this->channel_data_types_[i]<<"\n";
     }
  }
  os << indent << "Compression: " << this->compression_ << "\n";
  os << "\n";
  os << indent << "Planar configuration: " << this->PlanarConfiguration << "\n";
  os << indent << "Photometric interpretation: " << this->PhotometricInterpretation << "\n";
  os << indent << "Predictor: " << this->Predictor << "\n";
  os << indent << "Channel info:\n";

  for(int i=0;i<this->dimensions_[4];i++)
    {
        os << indent << indent << this->GetChannelName(i)<<",("<<this->GetChannelColorComponent(i,0)<<","<<this->GetChannelColorComponent(i,1)<<","<<this->GetChannelColorComponent(i,2)<<")\n";
    }
  os << indent << "Strip byte counts:\n";

  for(int i=0;i<this->dimensions_[4];i++)
    {
      os << indent << indent << this->strip_byte_count_[i] << "\n";
    }
}

std::unique_ptr<Image> LSMReader::read(ImageFactory* factory, const options_map&) {
    this->readHeader();

    const int dataType = this->GetDataTypeForChannel(0); // This could vary by channel!

    std::unique_ptr<Image> output = factory->create(
                            BYTES_BY_DATA_TYPE(dataType)*8,
                            this->dimensions_[2],
                            this->dimensions_[3],
                            this->dimensions_[4],
                            this->dimensions_[0],
                            this->dimensions_[1]
                            );

    byte* imstart = output->rowp_as<byte>(0);
    for (int z = 0; z < this->dimensions_[2]; ++z) {
        for (int timepoint = 0; timepoint < this->dimensions_[3]; ++timepoint) {
            for (int ch = 0; ch < this->dimensions_[4]; ++ch) {
                this->ConstructSliceOffsets(ch);
                byte* imdata = imstart + (z*(this->dimensions_[3]*this->dimensions_[4]) + timepoint*this->dimensions_[4] + ch)
                                    * this->dimensions_[0]*this->dimensions_[1]* BYTES_BY_DATA_TYPE(dataType);
                unsigned long offset = this->GetSliceOffset(timepoint, z);
                const int readSize = this->GetStripByteCount(timepoint, z);
                std::fill(imdata, imdata + readSize, 0);

                int bytes = ReadFile(this->src, &offset, readSize, imdata, true);

                if (bytes != readSize) {
                    throw ProgrammingError("Could not read data");
                }
                if (this->compression_ == LSM_COMPRESSED) {
                    this->DecodeLZWCompression(imdata, readSize, BYTES_BY_DATA_TYPE(dataType));
                }
            }
        }
    }
    return output;
}


} // namespace

std::unique_ptr<Image> LSMFormat::read(byte_source* s, ImageFactory* factory, const options_map& opts) {
    LSMReader reader(s);
    return reader.read(factory, opts);
}

