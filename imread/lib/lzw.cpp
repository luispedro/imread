// Copyright 2012 Luis Pedro Coelho
// This is an implementation of TIFF LZW compression coded from the spec.
// License: MIT

#include <vector>

struct code_stream {
    public:
        code_stream(unsigned char* buf, unsigned long len)
                :buf_(buf)
                ,byte_pos_(0)
                ,bit_pos_(0)
                ,len_(len)
        { }
        unsigned short getbit() {
            unsigned char val = buf_[byte_pos_];
            unsigned short res = (val & (1 << (8-bit_pos_)));
            ++bit_pos_;
            if (bit_pos_ == 8) {
                bit_pos_ = 0;
                ++byte_pos_;
                if (byte_pos_ > len_) {
                    throw CannotReadError("Unexpected End of File");
                }
            }
            return res;
        }
        unsigned short get(int nbits) {
            unsigned short res = 0;
            for (int i = 0; i != nbits; ++i) {
                res <<= 1;
                res |= getbit();
            }
            return res;
        }
    private:
        const unsigned char* buf_;
        int byte_pos_;
        int bit_pos_;
        const int len_;
};

std::string table_at(const std::vector<std::string> table, unsigned short index) {
    if (index < 256) {
        std::string res = "0";
        res[0] = (char)index;
        return res;
    }
    return table.at(index - 258);
}

void write_string(std::vector<unsigned char>& output, std::string s) {
    output.insert(output.end(), s.begin(), s.end());
}

std::vector<unsigned char> lzw_decode(void* buf, unsigned long len) {
    std::vector<std::string> table;
    std::vector<unsigned char> output;
    code_stream st(static_cast<unsigned char*>(buf), len);

    int nbits = 9;
    unsigned short old_code = 0;
    const short ClearCode = 256;
    const short EoiCode = 257;
    while (true) {
        const short code = st.get(nbits);
        if (code == EoiCode) break;
        if (code == ClearCode) {
            table.clear();
            nbits = 9;
            const short next_code = st.get(nbits);
            if (next_code == EoiCode) break;
            write_string(output, table[next_code]);
            old_code = next_code;
        } else if (code < 256 || (code - 258) < short(table.size())) {
            write_string(output, table_at(table,code));
            table.push_back(
                    table_at(table,old_code) + table_at(table,code)[0]
                    );
            old_code = code;
        } else {
            std::string out_string = table_at(table, old_code) + table_at(table, old_code)[0];
            write_string(output, out_string);
            table.push_back(out_string);
            if (table.size() == ( 512-258)) nbits = 10;
            if (table.size() == (1024-258)) nbits = 11;
            if (table.size() == (2048-258)) nbits = 12;
            old_code = code;
        }
   }
   return output;
}
