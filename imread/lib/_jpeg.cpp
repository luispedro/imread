// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#include "base.h"
#include "_jpeg.h"

#include <stdio.h>
#include <setjmp.h>

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

struct jpeg_dst_adaptor {
    jpeg_destination_mgr mgr;
    byte_sink* s;
    byte* buf;

    jpeg_dst_adaptor(byte_sink* s);

    ~jpeg_dst_adaptor() {
        delete [] buf;
    }
};

void nop(j_decompress_ptr cinfo) { }
void nop_dst(j_compress_ptr cinfo) { }

boolean fill_input_buffer(j_decompress_ptr cinfo) {
    jpeg_source_adaptor* adaptor = reinterpret_cast<jpeg_source_adaptor*>(cinfo->src);
    adaptor->mgr.next_input_byte = adaptor->buf;
    adaptor->mgr.bytes_in_buffer = adaptor->s->read(adaptor->buf, buffer_size);
    return true;
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

boolean empty_output_buffer(j_compress_ptr cinfo) {
    jpeg_dst_adaptor* adaptor = reinterpret_cast<jpeg_dst_adaptor*>(cinfo->dest);
    adaptor->s->write(adaptor->buf, buffer_size);
    adaptor->mgr.next_output_byte = adaptor->buf;
    adaptor->mgr.free_in_buffer = buffer_size;
    return TRUE;
}

void flush_output_buffer(j_compress_ptr cinfo) {
    jpeg_dst_adaptor* adaptor = reinterpret_cast<jpeg_dst_adaptor*>(cinfo->dest);
    adaptor->s->write(adaptor->buf, adaptor->mgr.next_output_byte - adaptor->buf);
}



jpeg_source_adaptor::jpeg_source_adaptor(byte_source* s)
    :s(s)
    {
        buf = new byte[buffer_size];
        mgr.next_input_byte = buf;
        mgr.bytes_in_buffer = 0;

        mgr.init_source = nop;
        mgr.fill_input_buffer = fill_input_buffer;
        mgr.skip_input_data = skip_input_data;
        mgr.resync_to_restart = jpeg_resync_to_restart;
        mgr.term_source = nop;
    }

jpeg_dst_adaptor::jpeg_dst_adaptor(byte_sink* s)
    :s(s)
    {
        buf = new byte[buffer_size];

        mgr.next_output_byte = buf;
        mgr.free_in_buffer = buffer_size;

        mgr.init_destination = nop_dst;
        mgr.empty_output_buffer = empty_output_buffer;
        mgr.term_destination = flush_output_buffer;
    }

struct jpeg_decompress_holder {
    jpeg_decompress_holder() { jpeg_create_decompress(&info); }
    ~jpeg_decompress_holder() { jpeg_destroy_decompress(&info); }

    jpeg_decompress_struct info;
};

struct jpeg_compress_holder {
    jpeg_compress_holder() { jpeg_create_compress(&info); }
    ~jpeg_compress_holder() { jpeg_destroy_compress(&info); }

    jpeg_compress_struct info;
};

struct error_mgr {
    error_mgr();

    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

void err_long_jump(j_common_ptr cinfo) {
  error_mgr* err = reinterpret_cast<error_mgr*>(cinfo->err);
  longjmp(err->setjmp_buffer, 1);
}


error_mgr::error_mgr() {
    jpeg_std_error(&pub);
    pub.error_exit = err_long_jump;
}

inline
J_COLOR_SPACE color_space(int components) {
    if (components == 1) return JCS_GRAYSCALE;
    if (components == 3) return JCS_RGB;
    throw CannotWriteError("unsupported image dimensions");
}

} // namespace

std::auto_ptr<Image> JPEGFormat::read(byte_source* src, ImageFactory* factory) {
    jpeg_source_adaptor adaptor(src);
    jpeg_decompress_holder c;

    // error management
    error_mgr jerr;
    c.info.err = &jerr.pub;

    // source
    c.info.src = &adaptor.mgr;

    if (setjmp(jerr.setjmp_buffer)) {
        throw CannotReadError();
    }
    // now read the header & image data
    jpeg_read_header(&c.info, TRUE);
    jpeg_start_decompress(&c.info);
    const int h = c.info.output_height;
    const int w = c.info.output_width;
    const int d = c.info.output_components;
    std::auto_ptr<Image> output(factory->create<byte>(h, w, d));

    for (int r = 0; r != h; ++r) {
        byte* rowp = output->rowp_as<byte>(r);
        jpeg_read_scanlines(&c.info, &rowp, 1);
    }
    jpeg_finish_decompress(&c.info);
    return output;
}


void JPEGFormat::write(Image* input, byte_sink* output) {
    jpeg_dst_adaptor adaptor(output);
    jpeg_compress_holder c;

    // error management
    error_mgr jerr;
    c.info.err = &jerr.pub;
    c.info.dest = &adaptor.mgr;

    if (setjmp(jerr.setjmp_buffer)) {
        throw CannotWriteError();
    }

    c.info.image_height = input->dim(0);
    c.info.image_width = input->dim(1);
    c.info.input_components = (input->ndims() > 2 ? input->dim(2) : 1);
    c.info.in_color_space = color_space(c.info.input_components);

    jpeg_set_defaults(&c.info);
    jpeg_start_compress(&c.info, TRUE);

    while (c.info.next_scanline < c.info.image_height) {
        JSAMPROW rowp = static_cast<JSAMPROW>(input->rowp_as<void>(c.info.next_scanline));
        (void) jpeg_write_scanlines(&c.info, &rowp, 1);
    }
    jpeg_finish_compress(&c.info);
}


