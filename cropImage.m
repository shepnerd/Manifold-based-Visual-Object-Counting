function [ range ] = cropImage( image_size, patch_size, step_size )


h = (floor((image_size(1)-patch_size+step_size)/step_size))*step_size+patch_size-step_size;
w = (floor((image_size(2)-patch_size+step_size)/step_size))*step_size+patch_size-step_size;

hs = floor((image_size(1)-h)/2)+1;
he = hs + h - 1;

ws = floor((image_size(2)-w)/2)+1;
we = ws + w - 1;

range.h = hs : he;
range.w = ws : we;

end

