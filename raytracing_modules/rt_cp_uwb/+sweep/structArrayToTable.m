function tbl = structArrayToTable(items)
% structArrayToTable - convert a cell array of scalar structs into a table.

    if isempty(items)
        tbl = table();
        return;
    end

    mask = ~cellfun(@isempty, items);
    if ~all(mask)
        items(~mask) = {struct()};
    end

    fields = {};
    for i = 1:numel(items)
        fields = union(fields, fieldnames(items{i}), 'stable');
    end

    n = numel(items);
    tbl = table();
    for f = 1:numel(fields)
        name = fields{f};
        values = cell(n, 1);
        for i = 1:n
            if isfield(items{i}, name)
                values{i} = items{i}.(name);
            else
                values{i} = [];
            end
        end
        tbl.(name) = collapseColumn(values);
    end
end

function col = collapseColumn(values)
    first_nonempty = find(~cellfun(@isempty, values), 1, 'first');
    if isempty(first_nonempty)
        col = values;
        return;
    end

    sample = values{first_nonempty};
    if (isnumeric(sample) || islogical(sample)) && all(cellfun(@(v) isempty(v) || (isnumeric(v) || islogical(v)) && isscalar(v), values))
        fill = NaN;
        if islogical(sample)
            fill = false;
        end
        col = repmat(fill, numel(values), 1);
        for i = 1:numel(values)
            if ~isempty(values{i})
                col(i) = values{i};
            end
        end
        if islogical(sample)
            col = logical(col);
        end
        return;
    end

    if ischar(sample) || isstring(sample)
        col = repmat({''}, numel(values), 1);
        for i = 1:numel(values)
            if ~isempty(values{i})
                col{i} = char(string(values{i}));
            end
        end
        return;
    end

    col = values;
end
