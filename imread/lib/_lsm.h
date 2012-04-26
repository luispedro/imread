/*=========================================================================

  Program:   BioImageXD
  Module:    $RCSfile: LSMFormat.h,v $
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
#ifndef __vtkLSMReader_h
#define __vtkLSMReader_h

// .NAME LSMFormat - read LSM files
// .SECTION Description
// LSMFormat is a source object that reads LSM files.
// It should be able to read most any LSM file
//
// .SECTION Thanks
// This class was developed as a part of the BioImageXD Project.
// The BioImageXD project includes the following people:
//
// Dan White <dan@chalkie.org.uk>
// Kalle Pahajoki <kalpaha@st.jyu.fi>
// Pasi Kankaanpää <ppkank@bytl.jyu.fi>
//
#include "base.h"

#include <vector>
#include <string>
#include <ostream>

class LSMFormat : public ImageFormat {
    public:
        LSMFormat();
        ~LSMFormat();

        bool can_read() const { return true; }

        std::auto_ptr<Image> read(byte_source* src, ImageFactory* factory);

    private:
        virtual void PrintSelf(std::ostream& os, const char* indent="");


        int GetHeaderIdentifier();
        int IsValidLSMFile();
        int IsCompressed();
        int GetNumberOfTimePoints();
        int GetNumberOfChannels();
        int OpenFile();

        int GetChannelColorComponent(int,int);
        std::string GetChannelName(int);
        void SetUpdateTimePoint(int);
        void SetUpdateChannel(int);

        void SetDataByteOrderToBigEndian();
        void SetDataByteOrderToLittleEndian();
        void SetDataByteOrder(int);
        int GetDataByteOrder();
        const char *GetDataByteOrderAsString();

        int GetDataTypeForChannel(unsigned int channel);
        unsigned int GetUpdateChannel();

    protected:

        int TIFF_BYTES(unsigned short);
        int BYTES_BY_DATA_TYPE(int);
        void Clean();
        unsigned long ReadImageDirectory(byte_source *,unsigned long);
        void SetChannelName(const char *,int);
        int ClearChannelNames();
        int FindChannelNameStart(const char *, int);
        int ReadChannelName(const char *, int, char *);
        int ReadChannelDataTypes(byte_source*, unsigned long);
        int ReadChannelColorsAndNames(byte_source *,unsigned long);
        int ReadTimeStampInformation(byte_source *,unsigned long);
        int ReadLSMSpecificInfo(byte_source *,unsigned long);
        int AnalyzeTag(byte_source *,unsigned long);
        int ReadScanInformation(byte_source*, unsigned long);
        int NeedToReadHeaderInformation();
        void NeedToReadHeaderInformationOn();
        void NeedToReadHeaderInformationOff();
        unsigned long SeekFile(int);
        unsigned long GetOffsetToImage(int, int);


        //void ExecuteData(vtkDataObject *out);
        void CalculateExtentAndSpacing(int extent[6],double spacing[3]);
        void DecodeHorizontalDifferencing(unsigned char *,int);
        void DecodeHorizontalDifferencingUnsignedShort(unsigned short*, int);
        void DecodeLZWCompression(unsigned  char *,int);
        void ConstructSliceOffsets();
        unsigned int GetStripByteCount(unsigned int timepoint, unsigned int slice);
        unsigned int GetSliceOffset(unsigned int timepoint, unsigned int slice);


        bool swap_bytes_;

        int IntUpdateExtent[6];
        unsigned long OffsetToLastAccessedImage;
        int NumberOfLastAccessedImage;
        int FileNameChanged;
        char *FileName;
        double VoxelSizes[3];
        int Dimensions[5];// x,y,z,time,channels
        int NumberOfIntensityValues[4];
        unsigned short Identifier;
        unsigned int NewSubFileType;
        std::vector<unsigned short> bits_per_sample_;
        unsigned int Compression;
        std::vector<unsigned int> strip_offset_;
        std::vector<unsigned int> channel_data_types_;
        std::vector<double> track_wavelengths_;
        unsigned int SamplesPerPixel;
        std::vector<unsigned int> strip_byte_count_;
        unsigned int LSMSpecificInfoOffset;
        unsigned short PhotometricInterpretation;
        unsigned long ColorMapOffset;
        unsigned short PlanarConfiguration;
        unsigned short Predictor;
        unsigned short ScanType;
        int DataScalarType;

        std::vector<unsigned int> image_offsets_;
        std::vector<unsigned int> read_sizes_;
        std::vector<double> detector_offset_first_image_;
        std::vector<double> detector_offset_last_image_;
        std::vector<std::string> laser_names_;

        double DataSpacing[3];
        int DataExtent[6];
        int NumberOfScalarComponents;
        int data_type_;
        unsigned long ChannelInfoOffset;
        unsigned long ChannelDataTypesOffset;
        std::vector<int> channel_colors_;
        std::vector<std::string> channel_names_;
        std::vector<double> time_stamp_info_;
        char* Objective;
        char* Description;
        double TimeInterval;
        byte_source* src;

    private:
        LSMFormat(const LSMFormat&);  // Not implemented.
        void operator=(const LSMFormat&);  // Not implemented.
};
#endif
