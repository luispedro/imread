/*=========================================================================

  Program:   BioImageXD
  Language:  C++

 This is an open-source copyright as follows:
 Copyright (c) 2004-2008 BioImageXD Development Team

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
#include "lzw.cpp"
#include <time.h>

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

#define VTK_FILE_BYTE_ORDER_BIG_ENDIAN 0
#define VTK_FILE_BYTE_ORDER_LITTLE_ENDIAN 1

#define PRT_EXT(ext) ext[0],ext[1],ext[2],ext[3],ext[4],ext[5]
#define PRT_EXT2(ext) ext[0]<<","<<ext[1]<<","<<ext[2]<<","<<ext[3]<<","<<ext[4]<<","<<ext[5]

#define CLEAR_CODE 256
#define EOI_CODE 257

namespace {

int ReadFile(byte_source* s, unsigned long *pos,int size,char *buf,bool swap=false)
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

int ReadData(byte_source* s, unsigned long *pos,int size,char *buf)
{
  return ReadFile(s,pos,size,buf,1);
}

unsigned char CharPointerToUnsignedChar(char *buf)
{
  return *((unsigned char*)(buf));
}

int CharPointerToInt(char *buf)
{
  return *((int*)(buf));
}

unsigned int CharPointerToUnsignedInt(char *buf)
{
  return *((unsigned int*)(buf));
}

short CharPointerToShort(char *buf)
{
  return *((short*)(buf));
}

unsigned short CharPointerToUnsignedShort(char *buf)
{
  return *((unsigned short*)(buf));
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

unsigned int ReadUnsignedInt(byte_source* s, unsigned long *pos)
{
  char buff[4];
  ReadFile(s,pos,4,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap4LE((unsigned int*)buff);
#endif
  return CharPointerToUnsignedInt(buff);
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

unsigned short ReadUnsignedShort(byte_source* s, unsigned long *pos)
{
  char buff[2];
  ReadFile(s,pos,2,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap2LE((unsigned short*)buff);
#endif
  return CharPointerToUnsignedShort(buff);
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


} // namespace

LSMFormat::LSMFormat()
{
  this->Objective = NULL;
  this->Clean();
}

LSMFormat::~LSMFormat()
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

void LSMFormat::Clean()
{
  this->IntUpdateExtent[0] = this->IntUpdateExtent[1] = this->IntUpdateExtent[2] = this->IntUpdateExtent[4] = 0;
  this->IntUpdateExtent[3] = this->IntUpdateExtent[5] = 0;

  this->DataExtent[0] = this->DataExtent[1] = this->DataExtent[2] = this->DataExtent[4] = 0;
  this->DataExtent[3] = this->DataExtent[5] = 0;
  this->OffsetToLastAccessedImage = 0;
  this->NumberOfLastAccessedImage = 0;
  this->FileNameChanged = 0;
  this->FileName = NULL;
  this->VoxelSizes[0] = this->VoxelSizes[1] = this->VoxelSizes[2] = 0.0;
  this->Identifier = 0;

  this->DataSpacing[0] = this->DataSpacing[1] = this->DataSpacing[2] =  1.0f;
  this->Dimensions[0] = this->Dimensions[1] = this->Dimensions[2] = this->Dimensions[3] = this->Dimensions[4] = 0;
  this->NewSubFileType = 0;
  this->bits_per_sample_.resize(4);
  this->Compression = 0;
  this->strip_offset_.resize(4);
  this->SamplesPerPixel = 0;
  this->strip_byte_count_.resize(4);
  this->Predictor = 0;
  this->PhotometricInterpretation = 0;
  this->PlanarConfiguration = 0;
  this->ColorMapOffset = 0;
  this->LSMSpecificInfoOffset = 0;
  this->NumberOfIntensityValues[0] = this->NumberOfIntensityValues[1] = this->NumberOfIntensityValues[2] = this->NumberOfIntensityValues[3] = 0;
  this->ScanType = 0;
  this->data_type_ = 0;
  if (this->Objective)
  {
  delete[] this->Objective;
  this->Objective = NULL;
  }
}

void LSMFormat::SetDataByteOrderToBigEndian()
{
#ifndef VTK_WORDS_BIGENDIAN
  this->swap_bytes_ = false;
#else
  this->swap_bytes_ = true;
#endif
}

void LSMFormat::SetDataByteOrderToLittleEndian()
{
#ifdef VTK_WORDS_BIGENDIAN
  this->swap_bytes_ = true;
#else
  this->swap_bytes_ = false;
#endif
}

void LSMFormat::SetDataByteOrder(int byteOrder)
{
  if ( byteOrder == VTK_FILE_BYTE_ORDER_BIG_ENDIAN )
    {
    this->SetDataByteOrderToBigEndian();
    }
  else
    {
    this->SetDataByteOrderToLittleEndian();
    }
}


std::string LSMFormat::GetChannelName(int chNum)
{
    if (chNum < 0 || chNum >= this->channel_names_.size()) return "";
    return this->channel_names_[chNum];
}

void LSMFormat::SetChannelName(const char * name, const int chNum)
{
    const int n_channels = this->GetNumberOfChannels();
    if(!name || chNum > n_channels) return;
    this->channel_names_.resize(n_channels);
    this->channel_names_[chNum] = std::string(name);
}

int LSMFormat::FindChannelNameStart(const char *nameBuff, int length)
{
  int i;
  char ch;
  for(i=0;i<length;i++)
    {
    ch = *(nameBuff+i);
    if(ch > 32)
      {
      break;
      }
    }
  return i;
}

int LSMFormat::ReadChannelName(const char *nameBuff, int length, char *buffer)
{
  int i;
  char component;
  for(i=0;i<length;i++)
    {
    component = *(nameBuff+i);
    *(buffer+i) = component;
    if(component == 0)
      {
      break;
      }
    }
  return i;
}

int LSMFormat::ReadChannelDataTypes(byte_source* s, unsigned long start)
{
    const unsigned int numOfChls = this->GetNumberOfChannels();
    this->channel_data_types_.resize(numOfChls);

    unsigned long pos = start;
    for(unsigned int i=0; i < numOfChls; i++) {
        this->channel_data_types_[i] = ReadUnsignedInt(s, &pos);
    }
    return 0;
}

int LSMFormat::ReadChannelColorsAndNames(byte_source* s, unsigned long start)
{
  int colNum,nameNum,sizeOfStructure,sizeOfNames,nameLength, nameSkip;
  unsigned long colorOffset,nameOffset,pos;
  char *nameBuff,*colorBuff,*name,*tempBuff;
  unsigned char component;

  pos = start;
  // Read size of structure
  sizeOfStructure = ReadInt(s,&pos);
  // Read number of colors
  colNum = ReadInt(s,&pos);
  // Read number of names
  nameNum = ReadInt(s,&pos);
  sizeOfNames = sizeOfStructure - ( (10*4) + (colNum*4) );

  nameBuff = new char[sizeOfNames+1];
  name = new char[sizeOfNames+1];
  colorBuff = new char[5];

  if(colNum != this->GetNumberOfChannels())
    {
    }
  if(nameNum != this->GetNumberOfChannels())
    {

    }

  // Read offset to color info
  colorOffset = ReadInt(s,&pos) + start;
  // Read offset to name info
  nameOffset = ReadInt(s,&pos) + start;

  /*
  this->channel_colors_->Reset();
  this->channel_colors_->SetNumberOfValues(3*(colNum+1));
  this->channel_colors_->SetNumberOfComponents(3);
  */


  // Read the colors
  for(int j = 0; j < this->GetNumberOfChannels(); j++)
    {
    ReadFile(s,&colorOffset,4,colorBuff,1);

    for(int i=0;i<3;i++)
      {
        component = CharPointerToUnsignedChar(colorBuff+i);

        this->channel_colors_[i + 3*j] = component;
      }
    }

  ReadFile(s,&nameOffset,sizeOfNames,nameBuff,1);

  nameLength = nameSkip = 0;
  tempBuff = nameBuff;
  for(int i = 0; i < this->GetNumberOfChannels(); i++)
    {
    nameSkip = this->FindChannelNameStart(tempBuff,sizeOfNames-nameSkip);
    nameLength = this->ReadChannelName(tempBuff+nameSkip,sizeOfNames-nameSkip,name);

    tempBuff += nameSkip + nameLength;
    this->SetChannelName(name,i);
    }

  delete [] nameBuff;
  delete [] name;
  delete [] colorBuff;
  return 0;
}

int LSMFormat::ReadTimeStampInformation(byte_source* s, unsigned long offset)
{
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
int LSMFormat::ReadLSMSpecificInfo(byte_source* s, unsigned long pos)
{
  unsigned long offset;

  pos += 2 * 4; // skip over the start of the LSMInfo
                // first 4 byte entry if magic number
                // second is number of bytes in this structure

  // Then we read X
  this->NumberOfIntensityValues[0] = ReadInt(s,&pos);

  // vtkByteSwap::Swap4LE((int*)&this->NumberOfIntensityValues[0]);
  this->Dimensions[0] = this->NumberOfIntensityValues[0];
  // Y
  this->NumberOfIntensityValues[1] = ReadInt(s,&pos);
  this->Dimensions[1] = this->NumberOfIntensityValues[1];
  // and Z dimension
  this->NumberOfIntensityValues[2] = ReadInt(s,&pos);
  this->Dimensions[2] = this->NumberOfIntensityValues[2];
  // Read number of channels
  this->Dimensions[4] = ReadInt(s,&pos);

  // Read number of timepoints
  this->NumberOfIntensityValues[3] = ReadInt(s,&pos);
  this->Dimensions[3] = this->NumberOfIntensityValues[3];

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
  this->ScanType = ReadShort(s,&pos);

  if (this->ScanType == 1)
	{
	  int tmp = this->Dimensions[1];
	  this->Dimensions[1] = this->Dimensions[2];
	  this->Dimensions[2] = tmp;
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
  offset = ReadUnsignedInt(s,&pos);
  this->ReadTimeStampInformation(s,offset);

  return 1;
}
int LSMFormat::ReadScanInformation(byte_source* s,  unsigned long pos)
{
    unsigned int entry, type, size;
    unsigned int subblocksOpen = 0;
    char* name;
    double gain;
    double wavelength;
    int mode;
    char* chName;
    int chIsOn = 0, trackIsOn = 0, isOn = 0;
    while( 1 ) {
        entry = ReadUnsignedInt(s, &pos);
        type =  ReadUnsignedInt(s, &pos);
        size =  ReadUnsignedInt(s, &pos);

        //printf("entry=%d\n", entry);
        if(type == TYPE_SUBBLOCK && entry == SUBBLOCK_END) subblocksOpen--;
        else if(type == TYPE_SUBBLOCK) {
            subblocksOpen++;
        }

        switch(entry) {
            case DETCHANNEL_ENTRY_DETECTOR_GAIN_FIRST:
                gain = ReadDouble(s, &pos);
                continue;
                break;
            case DETCHANNEL_ENTRY_DETECTOR_GAIN_LAST:
                gain = ReadDouble(s, &pos);
                continue;
                break;
            case DETCHANNEL_ENTRY_INTEGRATION_MODE:
                mode = ReadInt(s, &pos);
                continue;
                break;
            case LASER_ENTRY_NAME:
                name = new char[size+1];
                ReadData(s, &pos, size, name);
                //printf("Laser name: %s\n", name);

                //FIXME this->laser_names_->InsertNextValue(name);
                delete[] name;
                continue;
                break;
            case ILLUMCHANNEL_ENTRY_WAVELENGTH:
                wavelength = ReadDouble(s, &pos);

                continue;
                break;
            case ILLUMCHANNEL_DETCHANNEL_NAME:
                chName = new char[size+1];
                ReadData(s, &pos, size, chName);
//                printf("chName = %s\n", chName);
                delete[] chName;
                continue;
                break;
            case TRACK_ENTRY_ACQUIRE:
                trackIsOn = ReadInt(s, &pos);

                continue;
                break;
            case TRACK_ENTRY_NAME:
                chName = new char[size+1];
                ReadData(s, &pos, size, chName);
                if(trackIsOn) {
                  //  printf("Track name = %s is on\n", chName);
                }
                delete[] chName;
                continue;
                break;
            case DETCHANNEL_DETECTION_CHANNEL_NAME:
               chName = new char[size+1];
                ReadData(s, &pos, size, chName);
                if(chIsOn) {
                    //printf("Detection channel name = %s is on\n", chName);
                }
                delete[] chName;
                continue;
                break;
            case DETCHANNEL_ENTRY_ACQUIRE:
                chIsOn = ReadInt(s, &pos);
                continue;
                break;

            case ILLUMCHANNEL_ENTRY_AQUIRE:
                isOn = ReadInt(s, &pos);
                if(isOn) {
                   if(trackIsOn) {
                         //FIXME this->track_wavelengths_->InsertNextValue(wavelength);
                         //printf("Acquired using wavelength: %s\n", wavelength);
                   }
                }
                continue;
                break;
            case RECORDING_ENTRY_DESCRIPTION:
                Description = new char[size+1];
                ReadData(s, &pos, size, Description);
                //printf("Description: %s\n", Description);
                continue;
                break;
            case RECORDING_ENTRY_OBJETIVE:
				if (this->Objective)
				    {
					delete[] this->Objective;
					this->Objective = NULL;
					}
                Objective = new char[size+1];
                ReadData(s, &pos, size, Objective);
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
        if(subblocksOpen == 0) break;
        // By default, skip the entry
        pos += size;
    }
    return 0;
}
int LSMFormat::AnalyzeTag(byte_source* s, unsigned long startPos)
{
  unsigned short type,length,tag;
  unsigned long readSize;
  int value, dataSize,i;
  char tempValue[4],tempValue2[4];
  char *actualValue = NULL;
  tag = ReadUnsignedShort(s,&startPos);
  type = ReadUnsignedShort(s,&startPos);
  length = ReadUnsignedInt(s,&startPos);

  ReadFile(s,&startPos,4,tempValue);

  for(i=0;i<4;i++)tempValue2[i]=tempValue[i];
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap4LE((unsigned int*)tempValue2);
#endif
  value = CharPointerToUnsignedInt(tempValue2);

  // if there is more than 4 bytes in value,
  // value is an offset to the actual data
  dataSize = this->TIFF_BYTES(type);
  readSize = dataSize*length;
  if(readSize > 4 && tag != TIF_CZ_LSMINFO)
  {
    actualValue = new char[readSize];
    startPos = value;
   if(tag == TIF_STRIPOFFSETS ||tag == TIF_STRIPBYTECOUNTS) {
        if( !ReadFile(s,&startPos,readSize,actualValue) ) {
            throw "Failed to get strip offsets\n";
        }
    }
  }
  else
  {
      actualValue = new char[4];
      for(int o=0;o<4;o++)actualValue[o] = tempValue[o];
  }
  switch(tag)
  {
    case TIF_NEWSUBFILETYPE:
      this->NewSubFileType = value;
      break;

    case TIF_IMAGEWIDTH:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
#endif
      //this->Dimensions[0] = this->CharPointerToUnsignedInt(actualValue);
      //this->Dimensions[0] = value;
      break;

    case TIF_IMAGELENGTH:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
      //this->Dimensions[1] = this->CharPointerToUnsignedInt(actualValue);
#endif
      //this->Dimensions[1] = value;
      break;

    case TIF_BITSPERSAMPLE:
#ifdef VTK_WORDS_BIGENDIAN
        vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
        this->bits_per_sample_.resize(length);
        unsigned short bits_per_sample_;
        for(i=0;i<length;i++)
        {
           bits_per_sample_ = CharPointerToUnsignedShort(actualValue + (this->TIFF_BYTES(TIFF_SHORT)*i));
           this->bits_per_sample_[i] = bits_per_sample_;
        }
        break;

    case TIF_COMPRESSION:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->Compression = CharPointerToUnsignedShort(actualValue);
      break;

    case TIF_PHOTOMETRICINTERPRETATION:
#ifdef VTK_WORDS_BIGENDIAN
        vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
        this->PhotometricInterpretation = CharPointerToUnsignedShort(actualValue);
        break;

    case TIF_STRIPOFFSETS:
        this->strip_offset_.resize(length);
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LERange((unsigned int*)actualValue,length);
#endif
        if(length>1) {
            for(i=0;i<length;i++)
            {
                unsigned int* offsets = (unsigned int*)actualValue;
                this->strip_offset_[i] = offsets[i];
            }
        } else {
            this->strip_offset_[0] = value;
        }
        break;

    case TIF_SAMPLESPERPIXEL:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
#endif
      this->SamplesPerPixel = CharPointerToUnsignedInt(actualValue);
      break;

    case TIF_STRIPBYTECOUNTS:
#ifdef VTK_WORDS_BIGENDIAN
        vtkByteSwap::Swap4LERange((unsigned int*)actualValue,length);
#endif
        this->strip_byte_count_.resize(length);
        if (length > 1) {
            for(i=0;i<length;i++)
            {
                unsigned int* counts = (unsigned int*)actualValue;
                unsigned int bytecount = CharPointerToUnsignedInt(actualValue + (this->TIFF_BYTES(TIFF_LONG)*i));

                this->strip_byte_count_[i] = bytecount;
            }
        } else {
            this->strip_byte_count_[0] = value;
        }
        break;
    case TIF_PLANARCONFIGURATION:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->PlanarConfiguration = CharPointerToUnsignedShort(actualValue);
      break;
    case TIF_PREDICTOR:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->Predictor = CharPointerToUnsignedShort(actualValue);
      break;
    case TIF_COLORMAP:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
#endif
      //this->ColorMapOffset = CharPointerToUnsignedInt(actualValue);
      break;
    case TIF_CZ_LSMINFO:

      this->LSMSpecificInfoOffset = value;
      break;
    }

  if(actualValue)
    {
    delete [] actualValue;
    }
  return 0;
}


/*------------------------------------------------------------------------------------------*/

int LSMFormat::GetHeaderIdentifier()
{
  return this->Identifier;
}

int LSMFormat::IsValidLSMFile()
{
  if(this->GetHeaderIdentifier() == LSM_MAGIC_NUMBER) return 1;
  return 0;
}

int LSMFormat::IsCompressed()
{
  return (this->Compression == LSM_COMPRESSED ? 1 : 0);
}

int LSMFormat::GetNumberOfTimePoints()
{
  return this->Dimensions[3];
}

int LSMFormat::GetNumberOfChannels()
{
  return this->Dimensions[4];
}

unsigned int LSMFormat::GetStripByteCount(unsigned int timepoint, unsigned int slice) {
    return this->read_sizes_[timepoint * this->Dimensions[2] + slice];
}

unsigned int LSMFormat::GetSliceOffset(unsigned int timepoint, unsigned int slice) {
    return this->image_offsets_[timepoint * this->Dimensions[2] + slice];
}

void LSMFormat::ConstructSliceOffsets()
{
    unsigned long int startPos = 2;
    this->image_offsets_.resize(this->Dimensions[2] * this->Dimensions[3]);
    this->read_sizes_.resize(this->Dimensions[2] * this->Dimensions[3]);
    int channel = this->GetUpdateChannel();

    for(int tp = 0; tp < this->Dimensions[3]; tp++) {
        for(int slice = 0; slice < this->Dimensions[2]; slice++) {
            this->GetOffsetToImage(slice, tp);
            this->image_offsets_[tp * this->Dimensions[2] + slice] =  this->strip_offset_[channel];
            this->read_sizes_[tp * this->Dimensions[2] + slice] = this->strip_byte_count_[channel];
        }
    }
}

unsigned long LSMFormat::GetOffsetToImage(int slice, int timepoint)
{
  return this->SeekFile(slice+(timepoint*this->Dimensions[2]));
}

unsigned long LSMFormat::SeekFile(int image)
{
  unsigned long offset = 4, finalOffset;
  int readSize = 4,i=0;
  unsigned short numberOfTags = 0;
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

unsigned long LSMFormat::ReadImageDirectory(byte_source* s, unsigned long offset)
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


void LSMFormat::DecodeHorizontalDifferencing(unsigned char *buffer, int size)
{
  for(int i=1;i<size;i++)
    {
      *(buffer+i) = *(buffer+i) + *(buffer+i-1);
    }
}

void LSMFormat::DecodeHorizontalDifferencingUnsignedShort(unsigned short *buffer, int size)
{
  for(int i=1;i<size;i++)
    {
      *(buffer+i) = *(buffer+i) + *(buffer+i-1);
    }
}

void LSMFormat::DecodeLZWCompression(unsigned char* buffer, int size) {
    LZWState *s = new LZWState;

    unsigned char *outbuf = new unsigned char[size];

    unsigned char *outbufp = outbuf;
    unsigned char *bufp = buffer;

    int width = this->Dimensions[0];
    int channel = this->GetUpdateChannel();
    int bytes = this->BYTES_BY_DATA_TYPE(this->GetDataTypeForChannel(channel));
    int lines = size / (width*bytes);
    lzw_decode_init(s, 8, bufp, size);

    int decoded = lzw_decode(s, outbufp, size);
    outbufp = outbuf;
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
        buffer[i] = outbuf[i];
    }
    delete s;
    delete []outbuf;

}

int LSMFormat::GetDataTypeForChannel(unsigned int channel)
{
    if (this->data_type_) return this->data_type_;
    if (this->channel_data_types_.empty()) return 1;
    return this->channel_data_types_.at(channel);
}

//----------------------------------------------------------------------------
// Convert to Imaging API
std::auto_ptr<Image> LSMFormat::read(byte_source* s, ImageFactory* factory) {
    this->src = s;
    unsigned char *buf, *tempBuf;
    int size,readSize,numberOfPixels,timepoint,channel;
    int outExtent[6];

    this->ConstructSliceOffsets();

    // if given time point or channel index is bigger than maximum,
    // we use maximum
    timepoint = (this->IntUpdateExtent[3]>this->GetNumberOfTimePoints()-1?this->GetNumberOfTimePoints()-1:this->IntUpdateExtent[3]);
    channel = this->GetUpdateChannel();
    int nSlices = (outExtent[5]-outExtent[4])+1;
    numberOfPixels = this->Dimensions[0]*this->Dimensions[1]*(outExtent[5]-outExtent[4]+1 );
    int dataType = this->GetDataTypeForChannel(channel);
    size = numberOfPixels * this->BYTES_BY_DATA_TYPE(dataType);

    buf = new unsigned char[size];
    tempBuf = buf;

    for(int i=outExtent[4];i<=outExtent[5];i++)
    {
        unsigned long offset = this->GetSliceOffset(timepoint, i);
        readSize = this->GetStripByteCount(timepoint, i);
        for(int i=0;i<readSize;i++) tempBuf[i] = 0;

        int bytes = ReadFile(this->src, &offset, readSize, (char *)tempBuf, 1);

        if (bytes != readSize) {
            // FIXME this->src->clear();
        }
        if(this->IsCompressed())
        {
            this->DecodeLZWCompression(tempBuf,readSize);
        }
        tempBuf += readSize;
    }



/*
  vtkUnsignedCharArray *uscarray;
  vtkUnsignedShortArray *ussarray;
  if(this->BYTES_BY_DATA_TYPE(dataType) > 1) {
        ussarray = vtkUnsignedShortArray::New();
        ussarray->SetNumberOfComponents(1);
        ussarray->SetNumberOfValues(numberOfPixels);

        ussarray->SetArray((unsigned short *)buf, numberOfPixels, 0);
        data->GetPointData()->SetScalars(ussarray);

        ussarray->Delete();
  } else {
        uscarray = vtkUnsignedCharArray::New();
        uscarray->SetNumberOfComponents(1);
        uscarray->SetNumberOfValues(numberOfPixels);

        uscarray->SetArray(buf, numberOfPixels, 0);
        data->GetPointData()->SetScalars(uscarray);

        uscarray->Delete();
    }
    return 1;
    */
}
/*
int LSMFormat::RequestInformation ( char * outputVector)
{
  unsigned long startPos;
  unsigned int imageDirOffset;
  int dataType;


  char buf[12];

  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  this->SetDataByteOrderToLittleEndian();

  if(!this->NeedToReadHeaderInformation())
    {
    return 1;
    }


  if(!this->OpenFile())
    {
    this->Identifier = 0;
    return 0;
    }

  startPos = 2;  // header identifier

  this->Identifier = ReadUnsignedShort(this->src, &startPos);
  if(!this->IsValidLSMFile())
    {
    vtkErrorMacro("Given file is not a valid LSM-file.");
    return 0;
    }

  imageDirOffset = ReadUnsignedInt(this->src, &startPos);

  this->ReadImageDirectory(this->src, imageDirOffset);

  if(this->LSMSpecificInfoOffset)
    {
      ReadLSMSpecificInfo(this->src, (unsigned long)this->LSMSpecificInfoOffset);
    }
  else
    {
      vtkErrorMacro("Did not found LSM specific info!");
      return 0;
    }
  if( !(this->ScanType == 6 || this->ScanType == 0 || this->ScanType == 3 || this->ScanType == 1) )
    {
      vtkErrorMacro("Sorry! Your LSM-file must be of type 6 LSM-file (time series x-y-z) or type 0 (normal x-y-z) or type 3 (2D + time) or type 1 (x-z scan). Type of this File is " <<this->ScanType);
      return 0;
    }


  this->CalculateExtentAndSpacing(this->DataExtent,this->DataSpacing);
    outInfo->Set(vtkDataObject::SPACING(), this->DataSpacing, 3);


    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
    this->DataExtent, 6);

    this->NumberOfScalarComponents = 1;

  int channel = this->GetUpdateChannel();
  dataType = this->GetDataTypeForChannel(channel);
  if(dataType > 1)
    {
      this->DataScalarType = VTK_UNSIGNED_SHORT;
    }
  else
    {
	  this->DataScalarType = VTK_UNSIGNED_CHAR;
    }
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, this->DataScalarType,
  this->NumberOfScalarComponents);

  this->NeedToReadHeaderInformationOff();
  return 1;
}
*/

void LSMFormat::CalculateExtentAndSpacing(int extent[6],double spacing[3])
{
  extent[0] = extent[2] = extent[4] = 0;
  extent[1] = this->Dimensions[0] - 1;
  extent[3] = this->Dimensions[1] - 1;
  extent[5] = this->Dimensions[2] - 1;

  spacing[0] = int(this->VoxelSizes[0]*1000000);
  if (spacing[0] < 1.0) spacing[0] = 1.0;
  spacing[1] = this->VoxelSizes[1] / this->VoxelSizes[0];
  spacing[2] = this->VoxelSizes[2] / this->VoxelSizes[0];
}

//----------------------------------------------------------------------------

int LSMFormat::GetChannelColorComponent(int ch, int component)
{
    if (ch < 0 ||
        component < 0 ||
        component > 2 ||
        ch > this->GetNumberOfChannels()-1 ||
        ch >= this->channel_colors_.size()) return 0;
  return this->channel_colors_[(ch*3) + component];
}

void LSMFormat::SetUpdateTimePoint(int timepoint)
{
  if(timepoint < 0 || timepoint == this->IntUpdateExtent[3])
    {
    return;
    }
  this->IntUpdateExtent[3] = timepoint;
}

void LSMFormat::SetUpdateChannel(int ch)
{
  if(ch < 0 || ch == this->IntUpdateExtent[4])
    {
    return;
    }
  this->IntUpdateExtent[4] = ch;
}

void LSMFormat::NeedToReadHeaderInformationOn()
{
  this->FileNameChanged = 1;
}

void LSMFormat::NeedToReadHeaderInformationOff()
{
  this->FileNameChanged = 0;
}

int LSMFormat::NeedToReadHeaderInformation()
{
  return this->FileNameChanged;
}

int LSMFormat::BYTES_BY_DATA_TYPE(int type)
{
  int bytes = 1;
  switch(type)
    {
    case(1):
      return 1;
    case(2):
      return 2;
	case(3):
	  return 2;
    case(5):
      return 4;
    }
  return bytes;
}

int LSMFormat::TIFF_BYTES(unsigned short type)
{
  int bytes = 1;
  switch(type)
    {
    case(TIFF_BYTE):
      return 1;
    case(TIFF_ASCII):
    case(TIFF_SHORT):
      return 2;
    case(TIFF_LONG):
    case(TIFF_RATIONAL):
      return 4;
    }
  return bytes;
}

unsigned int LSMFormat::GetUpdateChannel() {
   return (this->IntUpdateExtent[4]>this->GetNumberOfChannels()-1?this->GetNumberOfChannels()-1:this->IntUpdateExtent[4]);

}

void LSMFormat::PrintSelf(std::ostream& os, const char* indent)
{
  os << indent << "Identifier: " << this->Identifier <<"\n";
  os << indent << "Dimensions: " << this->Dimensions[0] << "," << this->Dimensions[1] << ","<<this->Dimensions[2] << "\n";
  os << indent << "Time points: " << this->Dimensions[3] << "\n";
  os << "Number of channels: " << this->Dimensions[4] << "\n";
  os << "\n";
  os << indent << "Number of intensity values X: " << this->NumberOfIntensityValues[0] << "\n";
  os << indent << "Number of intensity values Y: " << this->NumberOfIntensityValues[1] << "\n";
  os << indent << "Number of intensity values Z: " << this->NumberOfIntensityValues[2] << "\n";
  os << indent << "Number of intensity values Time: " << this->NumberOfIntensityValues[3] << "\n";
  os << indent << "Voxel size X: " << this->VoxelSizes[0] << "\n";
  os << indent << "Voxel size Y: " << this->VoxelSizes[1] << "\n";
  os << indent << "Voxel size Z: " << this->VoxelSizes[2] << "\n";
  os << "\n";
  os << indent << "Scan type: " << this->ScanType << "\n";
  os << indent << "Data type: " << this->data_type_ << "\n";
  if(this->data_type_ == 0) {
     for(int i=0; i < this->GetNumberOfChannels(); i++) {
        os << indent << indent << "Data type of channel "<<i<<": "<< this->channel_data_types_[i]<<"\n";
     }
  }
  os << indent << "Compression: " << this->Compression << "\n";
  os << "\n";
  os << indent << "Planar configuration: " << this->PlanarConfiguration << "\n";
  os << indent << "Photometric interpretation: " << this->PhotometricInterpretation << "\n";
  os << indent << "Predictor: " << this->Predictor << "\n";
  os << indent << "Channel info:\n";

  for(int i=0;i<this->Dimensions[4];i++)
    {
        os << indent << indent << this->GetChannelName(i)<<",("<<this->GetChannelColorComponent(i,0)<<","<<this->GetChannelColorComponent(i,1)<<","<<this->GetChannelColorComponent(i,2)<<")\n";
    }
  os << indent << "Strip byte counts:\n";

  for(int i=0;i<this->Dimensions[4];i++)
    {
      os << indent << indent << this->strip_byte_count_[i] << "\n";
    }
}

