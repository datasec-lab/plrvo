function total_alpha = compute_alpha_sumplrvo_parallel(lambda, q, k, theta, N, C)
    chunk_size = 4;  % 1 million per chunk
    num_chunks = ceil(N / chunk_size);
    alpha_chunks = zeros(num_chunks, 1);

    eta = 0:(lambda + 1);
    n = lambda + 1;

    % Precompute log-binomial weights and constants
    log_binom = gammaln(n + 1) - gammaln(eta + 1) - gammaln(n - eta + 1);
    log_weights = log_binom + (n - eta) * log(1 - q) + eta * log(q);

    bk1 = eta ./ (2 * eta - 1);
    bk2 = (eta - 1) ./ (2 * eta - 1);

    alpha1 = C * (eta - 1) * theta;
    alpha2 = C * eta * theta;

    % Enable parallel processing if available
    parfor c = 1:num_chunks
        i_start = (c - 1) * chunk_size + 1;
        i_end = min(c * chunk_size, N);
        local_alpha = 0;

        for i = i_start:i_end
            gi = sqrt(i) - sqrt(i - 1);

            % Log-domain stable calculation
            log_f_minus = -k * log(1 - alpha1 * gi);
            log_f_plus  = -k * log(1 + alpha2 * gi);

            F_minus_term = log(bk1) + log_f_minus;
            F_plus_term  = log(bk2) + log_f_plus;

            max_logF = max(F_minus_term, F_plus_term);
            log_F_expr = max_logF + log(exp(F_minus_term - max_logF) + exp(F_plus_term - max_logF));

            exponents = log_weights + log_F_expr;
            max_exp = max(exponents);
            alpha_i = max_exp + log(sum(exp(exponents - max_exp)));

            local_alpha = local_alpha + alpha_i;
        end

        alpha_chunks(c) = local_alpha;
    end

    total_alpha = sum(alpha_chunks);
end