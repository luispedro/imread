// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#include "base.h"
#include "_jpeg.h"

#include <stdio.h>
extern "C" {
#include <jpeglib.h>
}

namespace {
const size_t buffer_size = 4096;

struct jpeg_source_adaptor {
    jpeg_source_mgr mgr;
    byte_source* s;
    byte* buf;
    
    jpeg_source_adaptor(byte_source* s);

    ~jpeg_source_adaptor() {
        delete [] buf;
    }
};

void init_source(j_decompress_ptr cinfo) {
}

boolean fill_input_buffer(j_decompress_ptr cinfo) {
    jpeg_source_adaptor* adaptor = reinterpret_cast<jpeg_source_adaptor*>(cinfo->src);
    adaptor->mgr.next_input_byte = adaptor->buf;
    adaptor->mgr.bytes_in_buffer = adaptor->s->read(adaptor->buf, buffer_size);
    return true;
}
void term_source(j_decompress_ptr cinfo) {
}

void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
    if (num_bytes <= 0) return;
    jpeg_source_adaptor* adaptor = reinterpret_cast<jpeg_source_adaptor*>(cinfo->src);
    while (num_bytes > long(adaptor->mgr.bytes_in_buffer)) {
        num_bytes -= adaptor->mgr.bytes_in_buffer;
        fill_input_buffer(cinfo);
    }
    adaptor->mgr.next_input_byte += num_bytes;
    adaptor->mgr.bytes_in_buffer -= num_bytes;
}

jpeg_source_adaptor::jpeg_source_adaptor(byte_source* s)
    :s(s)
    {
        buf = new byte[buffer_size];
        mgr.next_input_byte = buf;
        mgr.bytes_in_buffer = 0;

        mgr.init_source = init_source;
        mgr.fill_input_buffer = fill_input_buffer;
        mgr.skip_input_data = skip_input_data;
        mgr.resync_to_restart = jpeg_resync_to_restart;
        mgr.term_source = term_source;
    }
}

void JPEGFormat::read(byte_source* src, Image* output) {
    jpeg_source_adaptor adaptor(src);

    jpeg_decompress_struct cinfo;
    jpeg_create_decompress(&cinfo);

    // error management
    jpeg_error_mgr error;
    cinfo.err = jpeg_std_error(&error);

    // source
    cinfo.src = &adaptor.mgr;

    // now read the header & image data
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    const int h = cinfo.output_height;
    const int w = cinfo.output_width;
    const int d = cinfo.output_components;
    output->set_size(h, w, d);

    for (int r = 0; r != h; ++r) {
        byte* rowp = output->rowp_as<byte>(r);
        jpeg_read_scanlines(&cinfo, &rowp, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
}

