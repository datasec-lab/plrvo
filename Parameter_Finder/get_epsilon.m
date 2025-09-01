function eps_best = get_epsilon(T, clip, N, delta, q, theta, k)

% ==== Precompute gi vector ====
gi_vec = sqrt(1:N) - sqrt(0:(N - 1));

% Initialize best tracking
eps_best = Inf;
lambda_best = NaN;

% Start loop
lambda = 1;
lambda_thresh = 80;

% First total_alpha
total_alpha_prev = compute_alpha_sumplrvo_parallel(lambda, q, k, theta, N, clip);
rdp_prev = T * total_alpha_prev / (lambda );
 counter=0;
while lambda <= lambda_thresh
    total_alpha = compute_alpha_sumplrvo_parallel(lambda, q, k, theta, N, clip);
    rdp = T * total_alpha / lambda;

    % Full epsilon expression (Mironov bound)
    eps = rdp + log(lambda / (1 + lambda)) - ...
          (log(delta) + log(1 + lambda)) / lambda;

    % Track best epsilon
    if eps < eps_best
        eps_best = eps;
        lambda_best = lambda;

    else
        break
    end

    % Stop if RDP growth exceeds the rest of ε term growth
    rdp_diff = rdp - rdp_prev;

    % Difference in adjustment terms
    adj_prev = log((lambda - 1) / lambda) - (log(delta) + log(lambda)) / (lambda - 1);
    adj_curr = log(lambda / (lambda + 1)) - (log(delta) + log(lambda + 1)) / lambda;
    adj_diff = adj_curr - adj_prev;

    if rdp_diff > abs(adj_diff) 
        counter=counter+1;
    
    end

    rdp_prev = rdp;
    lambda = lambda + 1;
    if counter>1
        
        break;
    end
end
end
