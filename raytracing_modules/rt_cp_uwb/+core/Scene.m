classdef Scene < handle
    properties
        surfaces cell = {}
    end

    methods
        function obj = Scene(surfaces)
            if nargin > 0
                obj.surfaces = surfaces;
            end
        end

        function addSurface(obj, surface)
            obj.surfaces{end + 1} = surface;
        end
    end
end
