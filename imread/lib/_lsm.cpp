/*=========================================================================

  Program:   BioImageXD
  Module:    $RCSfile: vtkLSMReader.cxx,v $
  Language:  C++
  Date:      $Date: 2003/08/22 14:46:02 $
  Version:   $Revision: 1.39 $

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

#include "vtkLSMReader.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkSource.h"
#include "vtkPointData.h"
#include "vtkByteSwap.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include <time.h>


#define PRT_EXT(ext) ext[0],ext[1],ext[2],ext[3],ext[4],ext[5]
#define PRT_EXT2(ext) ext[0]<<","<<ext[1]<<","<<ext[2]<<","<<ext[3]<<","<<ext[4]<<","<<ext[5]

#define CLEAR_CODE 256
#define EOI_CODE 257
/*
 * LZW decoding
 * Copyright (c) 2003 Fabrice Bellard
 * Copyright (c) 2006 Konstantin Shishkov.
 * Licensed under LGPL, see Licenses/LGPL for full license
 */
#define LZW_MAXBITS                 12
#define LZW_SIZTABLE                (1<<LZW_MAXBITS)
struct LZWState {
    unsigned char *pbuf, *ebuf;
    int bbits;
    unsigned int bbuf;

    int cursize;                ///< The current code size
    int curmask;
    int codesize;
    int clear_code;
    int end_code;
    int newcodes;               ///< First available code
    int top_slot;               ///< Highest code for current size
    int extra_slot;
    int slot;                   ///< Last read code
    int fc, oc;
    unsigned char *sp;
    unsigned char stack[LZW_SIZTABLE];
    unsigned char suffix[LZW_SIZTABLE];
    unsigned short prefix[LZW_SIZTABLE];
    int bs;                     ///< current buffer size for GIF
};

int lzw_decode_init(LZWState *s, int csize, unsigned char *buf, int buf_size);
int lzw_decode(LZWState *s, unsigned char *buf, int len);


static const unsigned short mask[17] =
{
    0x0000, 0x0001, 0x0003, 0x0007,
    0x000F, 0x001F, 0x003F, 0x007F,
    0x00FF, 0x01FF, 0x03FF, 0x07FF,
    0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF
};

/* get one code from stream */
static int lzw_get_code(struct LZWState * s)
{
    int c;
        while (s->bbits < s->cursize) {
            s->bbuf = (s->bbuf << 8) | (*s->pbuf++);
            s->bbits += 8;
        }
        c = s->bbuf >> (s->bbits - s->cursize);
    s->bbits -= s->cursize;
    return c & s->curmask;
}


int lzw_decode_init(LZWState *p, int csize, unsigned char *buf, int buf_size)
{
    struct LZWState *s = (struct LZWState *)p;

    if(csize < 1 || csize > LZW_MAXBITS)
        return -1;
    /* read buffer */
    s->pbuf = buf;
    s->ebuf = s->pbuf + buf_size;
    s->bbuf = 0;
    s->bbits = 0;
    s->bs = 0;

    /* decoder */
    s->codesize = csize;
    s->cursize = s->codesize + 1;
    s->curmask = mask[s->cursize];
    s->top_slot = 1 << s->cursize;
    s->clear_code = 1 << s->codesize;
    s->end_code = s->clear_code + 1;
    s->slot = s->newcodes = s->clear_code + 2;
    s->oc = s->fc = -1;
    s->sp = s->stack;

    s->extra_slot = 1;
    return 0;
}

/**
 * Decode given number of bytes
 * NOTE: the algorithm here is inspired from the LZW GIF decoder
 *  written by Steven A. Bennett in 1987.
 */
int lzw_decode(LZWState *p, unsigned char *buf, int len){
    int l, c, code, oc, fc;
    unsigned char *sp;
    struct LZWState *s = (struct LZWState *)p;

    if (s->end_code < 0)
        return 0;

    l = len;
    sp = s->sp;
    oc = s->oc;
    fc = s->fc;

    for (;;) {
        while (sp > s->stack) {
            *buf++ = *(--sp);
            if ((--l) == 0)
                goto the_end;
        }
        c = lzw_get_code(s);
        if (c == s->end_code) {
            break;
        } else if (c == s->clear_code) {
            s->cursize = s->codesize + 1;
            s->curmask = mask[s->cursize];
            s->slot = s->newcodes;
            s->top_slot = 1 << s->cursize;
            fc= oc= -1;
        } else {
            code = c;
            if (code == s->slot && fc>=0) {
                *sp++ = fc;
                code = oc;
            }else if(code >= s->slot)
                break;
            while (code >= s->newcodes) {
                *sp++ = s->suffix[code];
                code = s->prefix[code];
            }
            *sp++ = code;
            if (s->slot < s->top_slot && oc>=0) {
                s->suffix[s->slot] = code;
                s->prefix[s->slot++] = oc;
            }
            fc = code;
            oc = c;
            if (s->slot >= s->top_slot - s->extra_slot) {
                if (s->cursize < LZW_MAXBITS) {
                    s->top_slot <<= 1;
                    s->curmask = mask[++s->cursize];
                }
            }
        }
    }
    s->end_code = -1;
  the_end:
    s->sp = sp;
    s->oc = oc;
    s->fc = fc;
    return len - l;
}

vtkStandardNewMacro(vtkLSMReader);

vtkLSMReader::vtkLSMReader()
{
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);      
  this->ChannelDataTypes = 0;
  this->TrackWavelengths = 0;
  this->ImageOffsets = 0;
  this->ReadSizes = 0;
  this->Objective = NULL;
  this->Clean();
}

vtkLSMReader::~vtkLSMReader()
{
  this->ClearFileName();
  this->ClearChannelNames();
  this->ChannelColors->Delete();
  this->BitsPerSample->Delete();
  this->StripOffset->Delete();
  this->StripByteCount->Delete();
  this->LaserNames->Delete();
  if(this->TrackWavelengths) {
    this->TrackWavelengths->Delete();
  }
  if(this->ChannelDataTypes) {
      this->ChannelDataTypes->Delete();
  }
  if(this->ImageOffsets) {
    this->ImageOffsets->Delete();
     this->ImageOffsets = 0;
     this->ReadSizes->Delete();
     this->ReadSizes = 0;
  }
  
}

void vtkLSMReader::ClearFileName()
{
  if (this->File)
    {
    this->File->close();
    delete this->File;
    this->File = NULL;
    }
  
  if (this->FileName)
    {
    delete [] this->FileName;
    this->FileName = NULL;
    }
}
//----------------------------------------------------------------------------
// This function sets the name of the file. 
void vtkLSMReader::SetFileName(const char *name)
{
  if ( this->FileName && name && (!strcmp(this->FileName,name)))
    {
    return;
    }
  if (!name && !this->FileName)
    {
    return;
    }
  if (this->FileName)
    {
    delete [] this->FileName;
    }
  if (name)
    {
    this->FileName = new char[strlen(name) + 1];
    strcpy(this->FileName, name);
    }
  else
    {
    this->FileName = NULL;
    }
  this->NeedToReadHeaderInformationOn();
  this->Modified();
}


void vtkLSMReader::Clean()
{
  this->IntUpdateExtent[0] = this->IntUpdateExtent[1] = this->IntUpdateExtent[2] = this->IntUpdateExtent[4] = 0;
  this->IntUpdateExtent[3] = this->IntUpdateExtent[5] = 0;
    
  this->DataExtent[0] = this->DataExtent[1] = this->DataExtent[2] = this->DataExtent[4] = 0;
  this->DataExtent[3] = this->DataExtent[5] = 0;    
  this->OffsetToLastAccessedImage = 0;
  this->NumberOfLastAccessedImage = 0;
  this->FileNameChanged = 0;
  this->FileName = NULL;
  this->File = NULL;
  this->VoxelSizes[0] = this->VoxelSizes[1] = this->VoxelSizes[2] = 0.0;
  this->Identifier = 0;
   
  this->LaserNames = vtkStringArray::New();
  this->TrackWavelengths = vtkDoubleArray::New();
  this->DataSpacing[0] = this->DataSpacing[1] = this->DataSpacing[2] =  1.0f;
  this->Dimensions[0] = this->Dimensions[1] = this->Dimensions[2] = this->Dimensions[3] = this->Dimensions[4] = 0;
  this->NewSubFileType = 0;
  this->BitsPerSample = vtkUnsignedShortArray::New();
  this->BitsPerSample->SetNumberOfTuples(4);
  this->BitsPerSample->SetNumberOfComponents(1);  
  this->Compression = 0;
  this->StripOffset = vtkUnsignedIntArray::New();
  this->StripOffset->SetNumberOfTuples(4);
  this->StripOffset->SetNumberOfComponents(1);  
  this->SamplesPerPixel = 0;
  this->StripByteCount = vtkUnsignedIntArray::New();
  this->StripByteCount->SetNumberOfTuples(4);
  this->StripByteCount->SetNumberOfComponents(1);  
  this->Predictor = 0;
  this->PhotometricInterpretation = 0;
  this->PlanarConfiguration = 0;
  this->ColorMapOffset = 0;
  this->LSMSpecificInfoOffset = 0;
  this->NumberOfIntensityValues[0] = this->NumberOfIntensityValues[1] = this->NumberOfIntensityValues[2] = this->NumberOfIntensityValues[3] = 0;
  this->ScanType = 0;
  this->DataType = 0;
  this->ChannelColors = vtkIntArray::New();
  this->ChannelNames = NULL;
  this->TimeStampInformation = vtkDoubleArray::New();
  if (this->Objective) 
  {
  delete[] this->Objective;
  this->Objective = NULL;
  }
}

int vtkLSMReader::OpenFile()
{
  if (!this->FileName)
    {
    vtkErrorMacro(<<"FileName must be specified.");
    return 0;
    }

  // Close file from any previous image
  if (this->File)
    {
    this->File->close();
    delete this->File;
    this->File = NULL;
    }
  
  // Open the new file
#ifdef _WIN32
  this->File = new ifstream(this->FileName, ios::in | ios::binary);
#else
  this->File = new ifstream(this->FileName, ios::in);
#endif
  if (! this->File || this->File->fail())
    {
    vtkErrorMacro(<< "OpenFile: Could not open file " <<this->FileName);
    return 0;
    }
  return 1;
}


void vtkLSMReader::SetDataByteOrderToBigEndian()
{
#ifndef VTK_WORDS_BIGENDIAN
  this->SwapBytesOn();
#else
  this->SwapBytesOff();
#endif
}

void vtkLSMReader::SetDataByteOrderToLittleEndian()
{
#ifdef VTK_WORDS_BIGENDIAN
  this->SwapBytesOn();
#else
  this->SwapBytesOff();
#endif
}

void vtkLSMReader::SetDataByteOrder(int byteOrder)
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

int vtkLSMReader::GetDataByteOrder()
{
#ifdef VTK_WORDS_BIGENDIAN
  if ( this->SwapBytes )
    {
    return VTK_FILE_BYTE_ORDER_LITTLE_ENDIAN;
    }
  else
    {
    return VTK_FILE_BYTE_ORDER_BIG_ENDIAN;
    }
#else
  if ( this->SwapBytes )
    {
    return VTK_FILE_BYTE_ORDER_BIG_ENDIAN;
    }
  else
    {
    return VTK_FILE_BYTE_ORDER_LITTLE_ENDIAN;
    }
#endif
}

const char *vtkLSMReader::GetDataByteOrderAsString()
{
#ifdef VTK_WORDS_BIGENDIAN
  if ( this->SwapBytes )
    {
    return "LittleEndian";
    }
  else
    {
    return "BigEndian";
    }
#else
  if ( this->SwapBytes )
    {
    return "BigEndian";
    }
  else
    {
    return "LittleEndian";
    }
#endif
}


const char* vtkLSMReader::GetChannelName(int chNum)
{
  if (!this->ChannelNames || chNum < 0 || chNum > this->GetNumberOfChannels()-1)
    {
    vtkDebugMacro(<<"GetChannelName: Illegal channel index!");
	return "";
    }
  return this->ChannelNames[chNum];
}

int vtkLSMReader::ClearChannelNames()
{
  vtkDebugMacro(<<"clearing " << this->GetNumberOfChannels()<<"channel names");
   if(!this->ChannelNames || this->GetNumberOfChannels() < 1)
    {
    return 0;
    }

  for (int i = 0; i < this->GetNumberOfChannels(); i++)
    {
    delete [] this->ChannelNames[i];
    }

  delete [] this->ChannelNames;
  vtkDebugMacro(<<"done");
  return 0;
}

int vtkLSMReader::AllocateChannelNames(int chNum)
{
  this->ClearChannelNames();
  vtkDebugMacro(<<"allocating space for "<<chNum<<"channel names");
  this->ChannelNames = new char*[chNum];
  if(!this->ChannelNames)
    {
    vtkErrorMacro(<<"Could not allocate memory for channel name table!");
    return 1;
    }
  for(int i=0;i<chNum;i++)
    {
    this->ChannelNames[i] = NULL;
    }
  return 0;
}

int vtkLSMReader::SetChannelName(const char *chName, int chNum)
{
  char *name;
  int length;
  if(!chName || chNum > this->GetNumberOfChannels())
    {
    return 0;
    }
  if(!this->ChannelNames)
    {
    this->AllocateChannelNames(this->GetNumberOfChannels());
    }
  
  length = strlen(chName);
  vtkDebugMacro(<<"length="<<length);    
  name = new char[length+1];
  if(!name)
    {
    vtkErrorMacro(<<"Could not allocate memory for channel name");
    return 1;
    }
  strncpy(name,chName,length);
  name[length] = 0;
  this->ChannelNames[chNum] = name;
  return 0;
}

int vtkLSMReader::FindChannelNameStart(const char *nameBuff, int length)
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
  if(i >= length)
    {
    vtkWarningMacro(<<"Start of the channel name may not be found!");
    }
  return i;
}

int vtkLSMReader::ReadChannelName(const char *nameBuff, int length, char *buffer)
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

int vtkLSMReader::ReadChannelDataTypes(ifstream *f,unsigned long start)
{
    unsigned long pos;
    unsigned int dataType; 
    pos = start;
    unsigned int numOfChls = this->GetNumberOfChannels();
    this->ChannelDataTypes = vtkUnsignedIntArray::New();
    this->ChannelDataTypes->SetNumberOfTuples(numOfChls);
    this->ChannelDataTypes->SetNumberOfComponents(1);  
    for(unsigned int i=0; i < numOfChls; i++) {
        dataType = this->ReadUnsignedInt(f, &pos);
        this->ChannelDataTypes->SetValue(i, dataType);
        vtkDebugMacro(<<"Channel "<<i<<" has datatype "<<dataType<<"\n");
    }
    return 0;
}

int vtkLSMReader::ReadChannelColorsAndNames(ifstream *f,unsigned long start)
{
  int colNum,nameNum,sizeOfStructure,sizeOfNames,nameLength, nameSkip;
  unsigned long colorOffset,nameOffset,pos;
  char *nameBuff,*colorBuff,*name,*tempBuff;
  unsigned char component;

  pos = start;
  // Read size of structure
  sizeOfStructure = this->ReadInt(f,&pos);
  vtkDebugMacro(<<"size of structure = "<<sizeOfStructure<<"\n");
  // Read number of colors
  colNum = this->ReadInt(f,&pos);
  // Read number of names
  nameNum = this->ReadInt(f,&pos);
  vtkDebugMacro(<<"nameNum="<<nameNum);
  sizeOfNames = sizeOfStructure - ( (10*4) + (colNum*4) );
  vtkDebugMacro(<<"sizeofNames="<<sizeOfNames<<"\n");

  nameBuff = new char[sizeOfNames+1];
  name = new char[sizeOfNames+1];
  colorBuff = new char[5];

  if(colNum != this->GetNumberOfChannels())
    {
    vtkDebugMacro(<<"Number of channel colors is not same as number of channels!");
    vtkDebugMacro(<<"numColors="<<colNum<<", numChls="<<this->GetNumberOfChannels()<<", numNames="<<nameNum);
    }
  if(nameNum != this->GetNumberOfChannels())
    {
    
    vtkDebugMacro(<<"Number of channel names is not same as number of channels!");
    vtkDebugMacro(<<"numColors="<<colNum<<", numChls="<<this->GetNumberOfChannels()<<", numNames="<<nameNum);
    }

  // Read offset to color info
  colorOffset = this->ReadInt(f,&pos) + start;
  // Read offset to name info
  nameOffset = this->ReadInt(f,&pos) + start;

  vtkDebugMacro(<<"colorOffset="<<colorOffset);
  vtkDebugMacro(<<"nameOffset="<<nameOffset);
  vtkDebugMacro(<<"number of colors"<< colNum);
  this->ChannelColors->Reset();
  this->ChannelColors->SetNumberOfValues(3*(colNum+1));
  this->ChannelColors->SetNumberOfComponents(3);

    
  // Read the colors
  for(int j = 0; j < this->GetNumberOfChannels(); j++)
    {
    this->ReadFile(f,&colorOffset,4,colorBuff,1);
    
    for(int i=0;i<3;i++)
      {
        component = this->CharPointerToUnsignedChar(colorBuff+i);        

        this->ChannelColors->SetValue(i+(3*j),component);
      }
    }

  this->ReadFile(f,&nameOffset,sizeOfNames,nameBuff,1);

  nameLength = nameSkip = 0;
  tempBuff = nameBuff;
  for(int i = 0; i < this->GetNumberOfChannels(); i++)
    {
    nameSkip = this->FindChannelNameStart(tempBuff,sizeOfNames-nameSkip);
    nameLength = this->ReadChannelName(tempBuff+nameSkip,sizeOfNames-nameSkip,name);
    
    tempBuff += nameSkip + nameLength;
    vtkDebugMacro(<<"Setting channel "<<i<<"name");
    this->SetChannelName(name,i);
    }
  
  delete [] nameBuff;
  delete [] name;
  delete [] colorBuff;
  return 0;
}

int vtkLSMReader::ReadTimeStampInformation(ifstream *f,unsigned long offset)
{
  int numOffStamps = 0;
  if( offset == 0 ) // position is 0 for non-timeseries files!
  {
    vtkDebugMacro(<<"No timestamp information available");
    return 0;
  }
  offset += 4;
  numOffStamps = this->ReadInt(f,&offset);
  vtkDebugMacro(<<"There are "<<numOffStamps<<" stamps available");
  if(numOffStamps != this->GetNumberOfTimePoints())
    {
//    vtkWarningMacro(<<"Number of time stamps does not correspond to the number off time points!");
    }
  this->TimeStampInformation->Reset();
  this->TimeStampInformation->SetNumberOfTuples(numOffStamps);
  this->TimeStampInformation->SetNumberOfComponents(1);
  for(int i=0;i<numOffStamps;i++)
    {
    this->TimeStampInformation->SetValue(i,this->ReadDouble(f,&offset));
    }
  return 0;
}

/* Read the TIF_CZ_LSMINFO entry described in Table 17 of the LSM file format specification
 *
 *
 */
int vtkLSMReader::ReadLSMSpecificInfo(ifstream *f,unsigned long pos)
{
  unsigned long offset;
  vtkDebugMacro("ReadLSMSpecificInfo(stream,"<<pos<<")\n");

  pos += 2 * 4; // skip over the start of the LSMInfo
                // first 4 byte entry if magic number
                // second is number of bytes in this structure

  // Then we read X
  this->NumberOfIntensityValues[0] = this->ReadInt(f,&pos); 
  
  // vtkByteSwap::Swap4LE((int*)&this->NumberOfIntensityValues[0]);
  this->Dimensions[0] = this->NumberOfIntensityValues[0];
  // Y
  this->NumberOfIntensityValues[1] = this->ReadInt(f,&pos); 
  this->Dimensions[1] = this->NumberOfIntensityValues[1];
  // and Z dimension
  this->NumberOfIntensityValues[2] = this->ReadInt(f,&pos); 
  this->Dimensions[2] = this->NumberOfIntensityValues[2];
  vtkDebugMacro(<<"Dimensions =" << Dimensions[0]<<","<<Dimensions[1]<<","<<Dimensions[2]<<"\n");
  // Read number of channels
  this->Dimensions[4] = this->ReadInt(f,&pos); 
  vtkDebugMacro(<<"Number of Channels"<<this->Dimensions[4]<<"\n");

  // Read number of timepoints
  this->NumberOfIntensityValues[3] = this->ReadInt(f,&pos);
  this->Dimensions[3] = this->NumberOfIntensityValues[3];

  // Read datatype, 1 for 8-bit unsigned int
  //                2 for 12-bit unsigned int
  //                5 for 32-bit float (timeseries mean of ROIs)
  //                0 if the channels have different types
  //                In that case, u32OffsetChannelDataTypes
  //                has further info
  this->DataType = this->ReadInt(f,&pos);
  vtkDebugMacro(<<"Data type="<<this->DataType<<"\n");

  // Skip the width and height of thumbnails
  pos += 2 * 4;

  // Read voxel sizes
  this->VoxelSizes[0] = this->ReadDouble(f,&pos);
  this->VoxelSizes[1] = this->ReadDouble(f,&pos);
  this->VoxelSizes[2] = this->ReadDouble(f,&pos);
  vtkDebugMacro("Voxel size="<<VoxelSizes[0]<<","<<VoxelSizes[1]<<","<<VoxelSizes[2]<<"\n");

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
  this->ScanType = this->ReadShort(f,&pos);
  vtkDebugMacro("ScanType="<<this->ScanType<<"\n");

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
  this->ChannelInfoOffset = this->ReadUnsignedInt(f,&pos);
  vtkDebugMacro(<<"Channel info offset (from addr"<<pos<<")="<<this->ChannelInfoOffset<<"\n");
  if (this->ChannelInfoOffset != 0)
	this->ReadChannelColorsAndNames(f,this->ChannelInfoOffset);

  // Skip time interval in seconds (8 bytes)
  //pos += 1*8;
  this->TimeInterval = this->ReadDouble(f, &pos);
  printf("Time interval = %f\n", this->TimeInterval);
  
  // If each channel has different datatype (meaning DataType == 0), then
  // read the offset to more information and read the info
  this->ChannelDataTypesOffset = this->ReadInt(f, &pos);
  unsigned long scanInformationOffset = this->ReadUnsignedInt(f, &pos);
  if(this->DataType == 0) {
    this->ReadChannelDataTypes(f, this->ChannelDataTypesOffset);
  }

  // Read scan information
  printf("Scan information offset = %d\n", scanInformationOffset);
  this->ReadScanInformation(f, scanInformationOffset);
  // SKip Zeiss Vision KS-3D speific data
  pos +=  4;
  // Read timestamp information
  offset = this->ReadUnsignedInt(f,&pos);
  this->ReadTimeStampInformation(f,offset);
  
  return 1;
}
int vtkLSMReader::ReadScanInformation(ifstream* f, unsigned long pos)
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
        entry = this->ReadUnsignedInt(f, &pos);
        type =  this->ReadUnsignedInt(f, &pos);
        size =  this->ReadUnsignedInt(f, &pos);
                
        //printf("entry=%d\n", entry);
        if(type == TYPE_SUBBLOCK && entry == SUBBLOCK_END) subblocksOpen--;
        else if(type == TYPE_SUBBLOCK) {
            subblocksOpen++;
        }
       
        switch(entry) {
            case DETCHANNEL_ENTRY_DETECTOR_GAIN_FIRST:
                gain = this->ReadDouble(f, &pos);
                continue;
                break;
            case DETCHANNEL_ENTRY_DETECTOR_GAIN_LAST:
                gain = this->ReadDouble(f, &pos);
                continue;
                break;
            case DETCHANNEL_ENTRY_INTEGRATION_MODE:
                mode = this->ReadInt(f, &pos);
                continue;
                break;
            case LASER_ENTRY_NAME:
                name = new char[size+1];
                this->ReadData(f, &pos, size, name);
                //printf("Laser name: %s\n", name);
                this->LaserNames->InsertNextValue(name);
                delete[] name;
                continue;
                break;
            case ILLUMCHANNEL_ENTRY_WAVELENGTH:
                wavelength = this->ReadDouble(f, &pos);
         
                continue;
                break;
            case ILLUMCHANNEL_DETCHANNEL_NAME: 
                chName = new char[size+1];
                this->ReadData(f, &pos, size, chName);
//                printf("chName = %s\n", chName);
                delete[] chName;
                continue;
                break;
            case TRACK_ENTRY_ACQUIRE:
                trackIsOn = this->ReadInt(f, &pos);
                
                continue;
                break;
            case TRACK_ENTRY_NAME:
                chName = new char[size+1];
                this->ReadData(f, &pos, size, chName);
                if(trackIsOn) {
                  //  printf("Track name = %s is on\n", chName);
                }
                delete[] chName;
                continue;
                break;      
            case DETCHANNEL_DETECTION_CHANNEL_NAME:
               chName = new char[size+1];
                this->ReadData(f, &pos, size, chName);
                if(chIsOn) {
                    //printf("Detection channel name = %s is on\n", chName);
                }
                delete[] chName;
                continue;
                break;
            case DETCHANNEL_ENTRY_ACQUIRE:
                chIsOn = this->ReadInt(f, &pos);
                continue;
                break;
    
            case ILLUMCHANNEL_ENTRY_AQUIRE:
                isOn = this->ReadInt(f, &pos);
                if(isOn) {
                   if(trackIsOn) {    
                         this->TrackWavelengths->InsertNextValue(wavelength);            
                         //printf("Acquired using wavelength: %f\n", wavelength);
                   }                   
                }
                continue;
                break;
            case RECORDING_ENTRY_DESCRIPTION:
                Description = new char[size+1];
                this->ReadData(f, &pos, size, Description);
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
                this->ReadData(f, &pos, size, Objective);
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
int vtkLSMReader::AnalyzeTag(ifstream *f,unsigned long startPos)
{
  unsigned short type,length,tag;
  unsigned long readSize;
  int value, dataSize,i;
  char tempValue[4],tempValue2[4];
  char *actualValue = NULL;
    //vtkDebugMacro(<<"Analyze tag start pos="<<startPos<<"\n");
  tag = this->ReadUnsignedShort(f,&startPos);
  type = this->ReadUnsignedShort(f,&startPos);
  length = this->ReadUnsignedInt(f,&startPos);
   
  this->ReadFile(f,&startPos,4,tempValue);

  for(i=0;i<4;i++)tempValue2[i]=tempValue[i];
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap4LE((unsigned int*)tempValue2);
#endif
  value = this->CharPointerToUnsignedInt(tempValue2);
  
  // if there is more than 4 bytes in value, 
  // value is an offset to the actual data
  dataSize = this->TIFF_BYTES(type);
  readSize = dataSize*length;
  if(readSize > 4 && tag != TIF_CZ_LSMINFO)
  {
    actualValue = new char[readSize];
    startPos = value;
   if(tag == TIF_STRIPOFFSETS ||tag == TIF_STRIPBYTECOUNTS) {
      // vtkDebugMacro(<<"Reading actual value from "<<startPos<<"to " << startPos+readSize);
        if( !this->ReadFile(f,&startPos,readSize,actualValue) ) {
            vtkErrorMacro(<<"Failed to get strip offsets\n");
            return 0;
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
//      vtkDebugMacro(<<"New subfile type="<<value);
      this->NewSubFileType = value;     
      break;
    
    case TIF_IMAGEWIDTH: 
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
      vtkDebugMacro(<<"Image width="<<value);
#endif
      //this->Dimensions[0] = this->CharPointerToUnsignedInt(actualValue);
      //this->Dimensions[0] = value;
      break;
    
    case TIF_IMAGELENGTH:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
      //this->Dimensions[1] = this->CharPointerToUnsignedInt(actualValue);
      vtkDebugMacro(<<"Image length="<<value);
#endif
      //this->Dimensions[1] = value;
      break;
    
    case TIF_BITSPERSAMPLE:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->BitsPerSample->SetNumberOfValues(length);
      unsigned short bitsPerSample;
      for(i=0;i<length;i++)
    {
      bitsPerSample = this->CharPointerToUnsignedShort(actualValue + (this->TIFF_BYTES(TIFF_SHORT)*i));
      this->BitsPerSample->SetValue(i,bitsPerSample);
    }
    break;
    
    case TIF_COMPRESSION:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->Compression = this->CharPointerToUnsignedShort(actualValue);
      break;
    
    case TIF_PHOTOMETRICINTERPRETATION:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->PhotometricInterpretation = this->CharPointerToUnsignedShort(actualValue);
      break;
    
    case TIF_STRIPOFFSETS:
      //      vtkDebugMacro(<<"Number of values="<<length);
      this->StripOffset->SetNumberOfValues(length);
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LERange((unsigned int*)actualValue,length);
#endif
    if(length>1) {
          for(i=0;i<length;i++)
        {
          unsigned int* offsets = (unsigned int*)actualValue;
          unsigned int stripOffset=offsets[i];
//          vtkDebugMacro(<<"Strip offset to "<<i<<"="<<stripOffset);   
          this->StripOffset->SetValue(i,stripOffset);
        }
    } else {
      //  vtkDebugMacro(<<"Strip offset to only channel="<<value);
        this->StripOffset->SetValue(0,value);
    }
      break;
    
    case TIF_SAMPLESPERPIXEL:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
#endif
      this->SamplesPerPixel = this->CharPointerToUnsignedInt(actualValue);
       //     vtkDebugMacro(<<"Samples per pixel="<<SamplesPerPixel<<"\n");
      break;
    
    case TIF_STRIPBYTECOUNTS:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LERange((unsigned int*)actualValue,length);
#endif      
      this->StripByteCount->SetNumberOfValues(length);

    if(length>1) {
          for(i=0;i<length;i++)
        {
          unsigned int* counts = (unsigned int*)actualValue;
          unsigned int bytecount = this->CharPointerToUnsignedInt(actualValue + (this->TIFF_BYTES(TIFF_LONG)*i));
          
            this->StripByteCount->SetValue(i,bytecount);
        //    vtkDebugMacro(<<"Strip byte count of " << i <<"="<<counts[i] <<"("<<bytecount<<")");
        }
    } else {
         //vtkDebugMacro(<<"Bytecount of only strip="<<value);
         this->StripByteCount->SetValue(0,value);
    }
      break;
    case TIF_PLANARCONFIGURATION:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->PlanarConfiguration = this->CharPointerToUnsignedShort(actualValue);
      break;
    case TIF_PREDICTOR:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap2LE((unsigned short*)actualValue);
#endif
      this->Predictor = this->CharPointerToUnsignedShort(actualValue);
      break;
    case TIF_COLORMAP:
#ifdef VTK_WORDS_BIGENDIAN
      vtkByteSwap::Swap4LE((unsigned int*)actualValue);
#endif
      //this->ColorMapOffset = this->CharPointerToUnsignedInt(actualValue);
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

int vtkLSMReader::GetHeaderIdentifier()
{  
  return this->Identifier;
}

int vtkLSMReader::IsValidLSMFile()
{
  if(this->GetHeaderIdentifier() == LSM_MAGIC_NUMBER) return 1;
  return 0;
}

int vtkLSMReader::IsCompressed()
{
  return (this->Compression == LSM_COMPRESSED ? 1 : 0);
}

int vtkLSMReader::GetNumberOfTimePoints()
{
  return this->Dimensions[3];
}

int vtkLSMReader::GetNumberOfChannels()
{
  return this->Dimensions[4];
}

unsigned int vtkLSMReader::GetStripByteCount(unsigned int timepoint, unsigned int slice) {
    if(!this->ReadSizes) {
        vtkErrorMacro(<<"Tried to get read size but table not constructed\n");    
    }
    return this->ReadSizes->GetValue(timepoint*this->Dimensions[2]+slice);
}   

unsigned int vtkLSMReader::GetSliceOffset(unsigned int timepoint, unsigned int slice) {
    if(!this->ImageOffsets) {
        vtkErrorMacro(<<"Request slice offset but table not constructed\n");
        return 0;
    }
    return this->ImageOffsets->GetValue(timepoint*this->Dimensions[2]+slice);
}

void vtkLSMReader::ConstructSliceOffsets()
{
    unsigned long int startPos = 2;
    if(!this->ImageOffsets) {
        this->ImageOffsets = vtkUnsignedIntArray::New();
        this->ReadSizes = vtkUnsignedIntArray::New();
        this->ImageOffsets->SetNumberOfTuples(this->Dimensions[2]*this->Dimensions[3]);
        this->ReadSizes->SetNumberOfTuples(this->Dimensions[2]*this->Dimensions[3]);
        for(int j = 0; j < this->Dimensions[2]*this->Dimensions[3]; j++) {
            this->ImageOffsets->SetValue(j, 0);
            this->ReadSizes->SetValue(j, 0);
        }
    }
  int channel = this->GetUpdateChannel();
  
  for(int tp = 0; tp < this->Dimensions[3]; tp++) {
    for(int slice = 0; slice < this->Dimensions[2]; slice++) {
        this->GetOffsetToImage(slice, tp);
        unsigned long offset = this->StripOffset->GetValue(channel);
        unsigned long readSize = this->StripByteCount->GetValue(channel);
        this->ImageOffsets->SetValue(tp*this->Dimensions[2]+slice, offset);
        this->ReadSizes->SetValue(tp*this->Dimensions[2]+slice, readSize);
    }
  }
}

unsigned long vtkLSMReader::GetOffsetToImage(int slice, int timepoint)
{
  return this->SeekFile(slice+(timepoint*this->Dimensions[2]));
}

unsigned long vtkLSMReader::SeekFile(int image)
{
  unsigned long offset = 4, finalOffset;
  int readSize = 4,i=0;
  unsigned short numberOfTags = 0;  
  int imageCount = image+1;

  if(this->OffsetToLastAccessedImage && (this->NumberOfLastAccessedImage < image))
  {
    offset = this->OffsetToLastAccessedImage;
    imageCount = image - this->NumberOfLastAccessedImage;
//    vtkDebugMacro(<<"offset of last image: "<<offset<<"imageCount="<<imageCount<<"\n");
  }
  else
  {
    offset = (unsigned long)this->ReadInt(this->GetFile(),&offset);
    vtkDebugMacro(<<"offset (from file): "<< offset<<"\n");
  }

  offset = this->ReadImageDirectory(this->GetFile(),offset);
//  vtkDebugMacro(<<"Offset: "<<offset<<", imageCount: "<<imageCount<<"\n");
  do
  {
    // we count only image directories and not thumbnail images
    // subfiletype 0 = images
    // subfiletype 1 = thumbnails
    //vtkDebugMacro(<<"SubFileType="<<this->NewSubFileType<<"\n");
    if(this->NewSubFileType == 0) 
    {
      i++;
    }
    finalOffset = offset;
    offset = this->ReadImageDirectory(this->GetFile(),offset);
    //vtkDebugMacro(<<"i="<<i<<", imageCount="<<imageCount<<", offset="<<offset<<"\n");
  } while(i<imageCount && offset != 0);

  this->OffsetToLastAccessedImage = finalOffset;
  this->NumberOfLastAccessedImage = image;

  return finalOffset;
}

unsigned long vtkLSMReader::ReadImageDirectory(ifstream *f,unsigned long offset)
{
  unsigned short numberOfTags=0;
  unsigned long nextOffset = offset;
  
  //vtkDebugMacro(<<"Reading unsigned short from "<<offset<<"\n");
  numberOfTags = this->ReadUnsignedShort(f,&offset);
  for(int i = 0; i < numberOfTags; i++)
  {   
    this->AnalyzeTag(f,offset);
    //vtkDebugMacro(<<"Tag analyed...\n");
    if(this->NewSubFileType == 1) {
      //vtkDebugMacro(<<"Found thumbnail, get next");
      break; //thumbnail image
    }
    offset = offset + 12;
    //vtkDebugMacro(<<"New offset="<<offset);
  }
  nextOffset += 2 + numberOfTags * 12;
  return this->ReadUnsignedInt(f,&nextOffset);
}


void vtkLSMReader::DecodeHorizontalDifferencing(unsigned char *buffer, int size)
{
  for(int i=1;i<size;i++)
    {
      *(buffer+i) = *(buffer+i) + *(buffer+i-1);
    }
}

void vtkLSMReader::DecodeHorizontalDifferencingUnsignedShort(unsigned short *buffer, int size)
{
  for(int i=1;i<size;i++)
    {
      *(buffer+i) = *(buffer+i) + *(buffer+i-1);
    }
}

void vtkLSMReader::DecodeLZWCompression(unsigned char* buffer, int size) {
    LZWState *s = new LZWState;
    
    unsigned char *outbuf = new unsigned char[size];
    
    unsigned char *outbufp = outbuf;
    unsigned char *bufp = buffer;
    
    int width = this->Dimensions[0];
    int channel = this->GetUpdateChannel();    
    int bytes = this->BYTES_BY_DATA_TYPE(this->GetDataTypeForChannel(channel));
    int lines = size / (width*bytes);
    lzw_decode_init(s, 8, bufp, size);
    vtkDebugMacro(<<"Size: "<<size<<", bytes per pixel: "<<bytes<<", lines: "<<lines<<", width: "<<width<<"\n");
    
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
    vtkDebugMacro(<<"Decoding done"<<"\n");
    delete s;
    delete []outbuf;
    
}

int vtkLSMReader::GetDataTypeForChannel(unsigned int channel)
{
   if(this->DataType != 0) {
    return this->DataType;
   }
   if(!this->ChannelDataTypes) return 1;
   return this->ChannelDataTypes->GetValue(channel);
}

//----------------------------------------------------------------------------
// Convert to Imaging API
int vtkLSMReader::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *outputVector)
{
  unsigned long offset, imageOffset;;
  unsigned char *buf, *tempBuf;
  int size,readSize,numberOfPixels,timepoint,channel;
  time_t start, end;
  int outExtent[6];

  if(!this->ImageOffsets) {
    vtkDebugMacro(<<"Constructing slice offset table\n");
    this->ConstructSliceOffsets();
  }
  
  // get the info object
  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkImageData *data = this->AllocateOutputData(outInfo->Get(vtkDataObject::DATA_OBJECT()));
  data->GetPointData()->GetScalars()->SetName("LSM Scalars");

    data->GetExtent(outExtent);  
  if(!this->Identifier)
    {
    vtkDebugMacro(<<"Can not execute data since information has not been executed!");
    return 0;
    }
  vtkDebugMacro(<<"Update extent: " << outExtent[0]<<", "<<outExtent[1]<<", "<<outExtent[2]<<", "<<outExtent[3]<<", "<<outExtent[4]<<"," << outExtent[5]<<"\n");
  // if given time point or channel index is bigger than maximum,
  // we use maximum
  timepoint = (this->IntUpdateExtent[3]>this->GetNumberOfTimePoints()-1?this->GetNumberOfTimePoints()-1:this->IntUpdateExtent[3]);
  channel = this->GetUpdateChannel();
  int nSlices = (outExtent[5]-outExtent[4])+1;
  vtkDebugMacro(<<"Timepoint="<<timepoint<<", channel="<<channel<<", "<<nSlices<<" slices"<<"\n");
  numberOfPixels = this->Dimensions[0]*this->Dimensions[1]*(outExtent[5]-outExtent[4]+1 );
  int dataType = this->GetDataTypeForChannel(channel);
  size = numberOfPixels * this->BYTES_BY_DATA_TYPE(dataType);
  vtkDebugMacro(<<"numberOfPixels=" << numberOfPixels << ", buffer size="<<size<<", datatype="<<dataType<<", bytes by datatype="<<this->BYTES_BY_DATA_TYPE(dataType)<<"\n");

  // this buffer will be deleted by the vtkXXXArray when the array is destroyed.
 
  buf = new unsigned char[size];
  tempBuf = buf;

  start = time (NULL);
  for(int i=outExtent[4];i<=outExtent[5];i++)
  {
    offset = this->GetSliceOffset(timepoint, i);
    readSize = this->GetStripByteCount(timepoint, i);

    vtkDebugMacro(<<"Offset to tp  "<<timepoint<<", slice " <<i<<" = "<<offset<<", strip byte count: " << readSize<<"\n");

    for(int i=0;i<readSize;i++)tempBuf[i] = 0;
    
    int bytes = this->ReadFile(this->GetFile(),&offset,readSize,(char *)tempBuf,1);
    
    if(bytes != readSize) {
        vtkDebugMacro(<<"Asked for " << readSize<<" bytes, got "<<bytes <<"\n");
        vtkDebugMacro(<<"File status: fail: "<<this->GetFile()->fail()<<", eof: "<<this->GetFile()->eof()<<"\n");
        this->GetFile()->clear();
    }
    if(this->IsCompressed())
    {
       this->DecodeLZWCompression(tempBuf,readSize);
    }
    tempBuf += readSize;
  }
  end = time (NULL);

  vtkDebugMacro(<<"Dataset generation time: "<<end-start);


  vtkUnsignedCharArray *uscarray;
  vtkUnsignedShortArray *ussarray;
  if(this->BYTES_BY_DATA_TYPE(dataType) > 1)
  {
    ussarray = vtkUnsignedShortArray::New();
    ussarray->SetNumberOfComponents(1);
    ussarray->SetNumberOfValues(numberOfPixels);
      
    ussarray->SetArray((unsigned short *)buf, numberOfPixels, 0);
    data->GetPointData()->SetScalars(ussarray);
    
    ussarray->Delete();
  }
  else
  {
    uscarray = vtkUnsignedCharArray::New();
    uscarray->SetNumberOfComponents(1);
    uscarray->SetNumberOfValues(numberOfPixels);
    
    uscarray->SetArray(buf, numberOfPixels, 0);
    data->GetPointData()->SetScalars(uscarray);
    
    uscarray->Delete();     
  }
 
    return 1;
  
}

int vtkLSMReader::RequestUpdateExtent (
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  int uext[6], ext[6];
    
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  //vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);

  outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext);
  // Get the requested update extent from the output.
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), uext);

  // If they request an update extent that doesn't cover the whole slice
  // then modify the uextent 
  if(uext[1] < ext[1] ) uext[1] = ext[1];
  if(uext[3] < ext[3] ) uext[3] = ext[3];
  outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), uext,6);
  //request->Set(vtkStreamingDemandDrivenPipeline::REQUEST_UPDATE_EXTENT(), uext,6);
  return 1;    
}

int vtkLSMReader::RequestInformation (

  vtkInformation       * vtkNotUsed( request ),

  vtkInformationVector** vtkNotUsed( inputVector ),

  vtkInformationVector * outputVector)
{
  unsigned long startPos;
  unsigned int imageDirOffset;
  int dataType;
  

  char buf[12];
  
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
   
  this->SetDataByteOrderToLittleEndian();

  if(!this->NeedToReadHeaderInformation())
    {
    vtkDebugMacro(<<"Don't need to read header information");
    return 1;
    }
    
  vtkDebugMacro(<<"Executing information.");

  if(!this->OpenFile())
    {
    this->Identifier = 0;
    return 0;
    }

  startPos = 2;  // header identifier

  this->Identifier = this->ReadUnsignedShort(this->GetFile(),&startPos);
  if(!this->IsValidLSMFile())
    {
    vtkErrorMacro("Given file is not a valid LSM-file.");
    return 0;
    }
  
  imageDirOffset = this->ReadUnsignedInt(this->GetFile(),&startPos);

  this->ReadImageDirectory(this->GetFile(),imageDirOffset);

  if(this->LSMSpecificInfoOffset)
    {        
      ReadLSMSpecificInfo(this->GetFile(),(unsigned long)this->LSMSpecificInfoOffset);
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
  
  vtkDebugMacro(<<"Executing information: first directory has been read.");
    
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
  vtkDebugMacro(<<"Executing information: executed.");
  return 1;
}

void vtkLSMReader::CalculateExtentAndSpacing(int extent[6],double spacing[3])
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

int vtkLSMReader::GetChannelColorComponent(int ch, int component)
{
  if (ch < 0 || ch > this->GetNumberOfChannels()-1 || component < 0 || component > 2 || ch >= this->ChannelColors->GetNumberOfTuples())
  {
    return 0;
  }
  return *(this->ChannelColors->GetPointer((ch*3) + component));
}

vtkImageData* vtkLSMReader:: GetTimePointOutput(int timepoint, int channel)
{
  this->SetUpdateTimePoint(timepoint);
  this->SetUpdateChannel(channel);
  return this->GetOutput();
}

void vtkLSMReader::SetUpdateTimePoint(int timepoint)
{
  if(timepoint < 0 || timepoint == this->IntUpdateExtent[3]) 
    {
    return;
    }
  this->IntUpdateExtent[3] = timepoint;
  this->Modified();
}

void vtkLSMReader::SetUpdateChannel(int ch)
{
  if(ch < 0 || ch == this->IntUpdateExtent[4])
    {
    return;
    }
  this->IntUpdateExtent[4] = ch;
  this->Modified();
}

void vtkLSMReader::NeedToReadHeaderInformationOn()
{
  this->FileNameChanged = 1;
}

void vtkLSMReader::NeedToReadHeaderInformationOff()
{
  this->FileNameChanged = 0;
}

int vtkLSMReader::NeedToReadHeaderInformation()
{
  return this->FileNameChanged;
}

ifstream *vtkLSMReader::GetFile()
{
  return this->File;
}

int vtkLSMReader::BYTES_BY_DATA_TYPE(int type)
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

int vtkLSMReader::TIFF_BYTES(unsigned short type)
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

unsigned char vtkLSMReader::CharPointerToUnsignedChar(char *buf)
{
  return *((unsigned char*)(buf));
}

int vtkLSMReader::CharPointerToInt(char *buf)
{
  return *((int*)(buf));
}

unsigned int vtkLSMReader::CharPointerToUnsignedInt(char *buf)
{
  return *((unsigned int*)(buf));
}

short vtkLSMReader::CharPointerToShort(char *buf)
{
  return *((short*)(buf));
}

unsigned short vtkLSMReader::CharPointerToUnsignedShort(char *buf)
{
  return *((unsigned short*)(buf));
}

double vtkLSMReader::CharPointerToDouble(char *buf)
{
  return *((double*)(buf));
}

int vtkLSMReader::ReadInt(ifstream *f,unsigned long *pos)
{
  char buff[4];
  this->ReadFile(f,pos,4,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap4LE((int*)buff);
#endif
  return CharPointerToInt(buff);
}

unsigned int vtkLSMReader::ReadUnsignedInt(ifstream *f,unsigned long *pos)
{
  char buff[4];
  this->ReadFile(f,pos,4,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap4LE((unsigned int*)buff);
#endif
  return this->CharPointerToUnsignedInt(buff);
}

short vtkLSMReader::ReadShort(ifstream *f,unsigned long *pos)
{
  char buff[2];
  this->ReadFile(f,pos,2,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap2LE((short*)buff);
#endif  
  return this->CharPointerToShort(buff);
}

unsigned short vtkLSMReader::ReadUnsignedShort(ifstream *f,unsigned long *pos)
{
  char buff[2];
  this->ReadFile(f,pos,2,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap2LE((unsigned short*)buff);
#endif
  return this->CharPointerToUnsignedShort(buff);
}

double vtkLSMReader::ReadDouble(ifstream *f,unsigned long *pos)
{
  char buff[8];
  this->ReadFile(f,pos,8,buff);
#ifdef VTK_WORDS_BIGENDIAN
  vtkByteSwap::Swap8LE((double*)buff);
#endif  
  return this->CharPointerToDouble(buff);
}

int vtkLSMReader::ReadData(ifstream *f,unsigned long *pos,int size,char *buf)
{
  return this->ReadFile(f,pos,size,buf,1);
}

int vtkLSMReader::ReadFile(ifstream *f,unsigned long *pos,int size,char *buf,bool swap)
{
  unsigned int ret = 1;
  f->seekg(*pos,ios::beg);
  f->read(buf,size);
#ifdef VTK_WORDS_BIGENDIAN
  if(swap) {
    vtkByteSwap::SwapLERange(buf,size);
  }      
#endif
  //if(f->fail() || f->eof()) ret=0;
  ret = f->gcount();
  if( !f ) return 0;
  *pos = *pos + size;
  return ret;
}

unsigned int vtkLSMReader::GetUpdateChannel() {
   return (this->IntUpdateExtent[4]>this->GetNumberOfChannels()-1?this->GetNumberOfChannels()-1:this->IntUpdateExtent[4]);

}

void vtkLSMReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
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
  os << indent << "Data type: " << this->DataType << "\n";
  if(this->DataType == 0) {
     for(int i=0; i < this->GetNumberOfChannels(); i++) {
        os << indent << indent << "Data type of channel "<<i<<": "<< this->ChannelDataTypes->GetValue(i)<<"\n";
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
      os << indent << indent << this->StripByteCount->GetValue(i) << "\n";
    }
}

