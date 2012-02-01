#ifndef LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
class PNGFormat : public ImageFormat {
    public:
        bool can_read() const { return true; }
        bool can_write() const { return true; }

        void read(byte_source* src, Image* output);
        //void write(Image* input, byte_sink* output);
};


#endif // LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
