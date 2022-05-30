#pragma once
#include <IO/ReadBufferFromFileDecorator.h>


namespace DB
{

class BoundedReadBuffer : public ReadBufferFromFileDecorator
{
public:
    explicit BoundedReadBuffer(std::unique_ptr<SeekableReadBuffer> impl_);

    bool supportsRightBoundedReads() const override { return true; }

    void setReadUntilPosition(size_t position) override;

    void setReadUntilEnd() override;

    bool nextImpl() override;

    off_t seek(off_t off, int whence) override;

    Range getRemainingReadRange() const override;

    size_t getFileOffsetOfBufferEnd() const override { return file_offset_of_buffer_end; }

    /// file_offset_of_buffer_end can differ from impl's file_offset_of_buffer_end
    /// becuase of resizing of the tail. => Need to also override getPosition() as
    /// it uses file_offset_of_buffer_end.
    off_t getPosition() override;

private:
    std::optional<size_t> read_until_position;
    size_t file_offset_of_buffer_end = 0;
};

}
