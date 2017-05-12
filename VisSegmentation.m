function [ ] = VisSegmentation( I, SEG )
%VISSEGMENTATION Summary of this function goes here
%   Detailed explanation goes here
perim = true(size(I,1), size(I,2));
for k = 1 : max(SEG(:))
    regionK = SEG == k;
    perimK = bwperim(regionK, 8);
    perim(perimK) = false;
end
perim = uint8(cat(3,perim,perim,perim));
finalImage = I .* perim;
imshow(finalImage);

end

