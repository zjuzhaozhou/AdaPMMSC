function [P] = ProSimplex(C)
    b = sort (C, 'descend');
    nAB = size(C,2);
    sum = 0;
    rho = nAB;
    for t = 1 : nAB
        sum = sum + b(t);
        if b(t) + 1/t*(1 - sum) <= 0
            rho = t - 1;
            sum = sum - b(t);
            break;
        end
    end
    z = 1/rho*(1-sum);
    P = (C + z > 0).*(C + z);
end