function [x_star, y_star, s_star, fval] = InteriorMethod(c, A, b, Aeq, beq, lb, ub, denseIndex, options)
% ���_�@�̎���
% options�ő���
[A_trans, b_trans, c_trans] = transform(c, A, b, Aeq, beq, lb, ub);

[iter, isPrimalFeasible, isDualFeasible, x_star, y_star, s_star] = hsdipm(A_trans, b_trans, c_trans, denseIndex);
fval = full(dot(c_trans, x_star));

% ���̓���
[~, n] = size(A);
x_star = x_star(1:n);
if 0
	fprintf('�œK����x =');fprintf('%f, ', x_star);fprintf('\n');
end
end



function [A_trans, b_trans, c_trans] = transform(c, A, b, Aeq, beq, lb, ub)
[m_ineq, n] = size(A);
[m_eq, ~] = size(Aeq);
lb_size = size(lb, 1);
ub_size = size(ub, 1);

[row,col,v] = find([Aeq; A]);

row_tmp = zeros(2*lb_size + 2*ub_size + m_ineq, 1);
col_tmp = zeros(2*lb_size + 2*ub_size + m_ineq, 1);
v_tmp = zeros(2*lb_size + 2*ub_size + m_ineq, 1);

for i = 1:lb_size
	row_tmp(i) = m_eq+m_ineq+i;
	col_tmp(i) = i;
	v_tmp(i) = -1;
	
	row_tmp(lb_size+i) = m_eq+m_ineq+i;
	col_tmp(lb_size+i) = n+m_ineq+i;
	v_tmp(lb_size+i)=1;
end
for i = 1:ub_size
	row_tmp(2*lb_size+i) = m_eq+m_ineq+n+i;
	col_tmp(2*lb_size+i) = i;
	v_tmp(2*lb_size+i) = 1;
	
	row_tmp(2*lb_size+ub_size+i) = m_eq+m_ineq+n+i;
	col_tmp(2*lb_size+ub_size+i) = n+m_ineq+n+i;
	v_tmp(2*lb_size+ub_size+i)=1;
end
for i = 1:m_ineq
	row_tmp(2*lb_size+2*ub_size+i) = m_eq+i;
	col_tmp(2*lb_size+2*ub_size+i) = n+i;
	v_tmp(2*lb_size+2*ub_size+i)=1;
end

row = [row; row_tmp]; col = [col; col_tmp]; v = [v; v_tmp];

A_trans = sparse(row, col, v);


b_trans = [beq; b; -lb; ub];
b_trans = sparse(b_trans);

[row,col,v] = find(c);
c_trans = sparse(row, col, v, n+m_ineq+lb_size+ub_size, 1);
end



function [iter, isPrimalFeasible, isDualFeasible, x_star, y_star, s_star] = hsdipm(A, b, c, denseIndex)
% homogeneous self-dual Interior Point Method
% �p�����[�^�ݒ�
beta     = 0.1;     % ���S�p�X�̎�����
gamma    = 0.9;     % �X�e�b�v�T�C�Y�̊���
epsilon  = 1.0e-3;
lambda   = 1e+2;    % �����_�ŗp����
alphaMin = 1.04-4;  % alpha�̉���, alpha�̓X�e�b�v��
auxMIn   = 1.0e-6;  % kappa, tau�̉���
maxIter  = 15;    % ���_�@�̍ő唽����

zeta = 1; % Write�̃e�L�X�g��1
[m, n] = size(A);

%�����_�ݒ�
e     = ones(n, 1);
x     = lambda * e;
y     = zeros(m, 1);
s     = lambda * e;
kappa = zeta;
tau   = 1;
theta = 1;

%�����_�Ɋ�Â��x�N�g��
q_p = b - A*x;
q_d = c - A.'*y - s;
q_c = dot(c,x) - dot(b,y) + zeta;
q_zeta = dot(x,s) + zeta;

iter = 0;
alpha = 0;
alpha_bar = 0;

%fprintf('it: cx     by     rp     rd     mu      alpha\n');

while iter < maxIter
	% ��o�΂̉��̎c���v�Z
	x_star = x / tau;
	y_star = y / tau;
	s_star = s / tau;
	r_p_star = b - A * x_star; % ����̎c��
	r_d_star = c - A.'*y_star - s_star; % �o�Ζ��̎c��
%	fprintf('%d: %.2f  %.2f  %.2f  %.2f  %.2f  %.2f \n', ...
%		iter, dot(c,x_star), dot(b,y_star), norm(r_p_star)/norm(b), norm(r_d_star)/norm(c), dot(x_star,s_star)/n, alpha);
	
	% �c�����������Ȃ�����I��
	if max([norm(r_p_star)/norm(b), norm(r_d_star)/norm(c), dot(x_star,s_star)/n]) < epsilon
		break;
	end
	
	% mu�̌v�Z
	mu = (dot(x,s) + kappa*tau)/(n+1);
	
	% �X�p�[�X�ȑΊp�s��̕ێ�
	n_lst = linspace(1, n, n);
	diag_x = sparse(n_lst, n_lst, x);
	diag_s_inv = sparse(n_lst, n_lst, s)^-1;
	
	% �T�������̂��߂̃x�N�g���ݒ�
	r_p	 = -A*x + b*tau - q_p*theta;
	r_d	 = A.'*y + s - c*tau + q_d*theta;
	r_c	 = dot(c,x) - dot(b,y) + kappa - q_c*theta;
	r_zeta = -q_zeta - dot(q_d,x) + dot(q_p, y) + q_c*tau;
	r_u	 = beta*mu*e - diag_x*s;
	r_v	 = beta*mu - tau*kappa;
	
	tau_zeta = - r_zeta / q_c;
	tau_c		= q_d / q_c;
	tau_b		= - q_p / q_c;
	
	theta_zeta = (tau*r_c + r_v - kappa*tau_zeta) / (q_c * tau);
	theta_c = (tau*c - kappa*tau_c) / (q_c * tau);
	theta_b = (-tau*b - kappa*tau_b) / (q_c * tau);
	
    %{
	% schur�̕⊮������������
	U_dense = zeros(m, denseIndex);
	V_dense = zeros(m, denseIndex);
    %}

    %{
	ind = 1;
	for i = 1:denseIndex
		U_dense(:, ind) = sparse(diag_x(i,i)*diag_s_inv(i,i)*A(:, i));
		V_dense(:, ind) = sparse(A(:, i));
		ind = ind + 1;
    end
    %}
    
    U_dense = A(:, 1:denseIndex) * diag_x(1:denseIndex, 1:denseIndex)*diag_s_inv(1:denseIndex,1:denseIndex);
    V_dense = A(:, 1:denseIndex);
    
    
	G_bar = A(:, denseIndex+1:n)*diag_x(denseIndex+1:n, denseIndex+1:n)* ...
                        diag_s_inv(denseIndex+1:n, denseIndex+1:n)*A(:, denseIndex+1:n).';
                    
	B_u1 = solveSchur(A, G_bar, U_dense, V_dense, diag_x, diag_s_inv, ...
						-b, c, zeros(n,1), denseIndex);
	B_u2 = solveSchur(A, G_bar, U_dense, V_dense, diag_x, diag_s_inv, ...
						q_p, -q_d, zeros(n,1), denseIndex);
	B_r  = solveSchur(A, G_bar, U_dense, V_dense, diag_x, diag_s_inv, ...
						r_p + tau_zeta * b - theta_zeta * q_p, ...
						r_d - tau_zeta * c + theta_zeta * q_d, r_u, denseIndex);
    
	V = sparse([tau_c, theta_c; tau_b, theta_b; zeros(n, 1), zeros(n, 1)]);
	B_U = [B_u1, B_u2];
	
	% �T�C�Y�m�ۂ̂��߁C�㔼��������v�Z
	delta_hat_x = B_r - B_U*((eye(2) + V.'*B_U)^-1*V.'*B_r); 
	
	% �T�������̌v�Z
	delta_x = delta_hat_x(1 : n);
	delta_y = delta_hat_x(n+1 : n+m);
	delta_s = delta_hat_x(n+m+1 : n+m+n);	
	delta_tau = tau_zeta + dot(tau_c, delta_x) + dot(tau_b, delta_y);
	delta_theta = theta_zeta + dot(theta_c, delta_x) + dot(theta_b, delta_y);
	delta_kappa = r_v / tau - kappa / tau * delta_tau;
	
	% alpha�̐ݒ�
	bar_alpha_p = 100;
	bar_alpha_d = 100;
	bar_alpha_kappa = 100;
	bar_alpha_tau = 100;
	if min(delta_x) > 0
		bar_alpha_p = 1.0;
	else
		for i = 1:max(size(delta_x))
			if delta_x(i) < 0 && bar_alpha_p > -x(i)/delta_x(i)
				bar_alpha_p = -x(i)/delta_x(i);
			end
		end
	end
	if min(delta_s) > 0
		bar_alpha_d = 1.0;
	else
		for i = 1:max(size(delta_s))
			if delta_s(i) < 0 && bar_alpha_d > -s(i)/delta_s(i)
				bar_alpha_d = -s(i)/delta_s(i);
			end
		end
	end
	if delta_kappa > 0
		bar_alpha_kappa = 1.0;
	else
		bar_alpha_kappa = - kappa/delta_kappa;
	end
	if delta_tau > 0
		bar_alpha_tau = 1.0;
	else
		bar_alpha_tau = - tau/delta_tau;
	end	
	alpha_bar = min([gamma*bar_alpha_p, gamma*bar_alpha_d,gamma*bar_alpha_kappa, gamma*bar_alpha_tau,1.0]);
	
	% neigbor�̓���
	max_neibor_iter = 1000;
	n_gamma = 0.5;
	eta = 0.7; % �ߖT�̍L�� (1�Ȃ�Β��S�p�X�Ɠ�����)
	for l = 0:max_neibor_iter
		alpha = n_gamma^l * alpha_bar;
		next_iter = dot(x+alpha*delta_x, s+alpha*delta_s) + (tau+alpha*delta_tau)*(kappa+alpha*delta_kappa);
		mu = next_iter / (n+1);
		if next_iter >= eta*mu
			break;
		end
	end
%	fprintf('���� = %d, next_iter = %f, mu=%f\n', l, next_iter, eta*mu);
	if alpha < alphaMin
		break;
	end
	x		= x + alpha*delta_x;
	y		= y + alpha*delta_y;
	s		= s + alpha*delta_s;
	kappa   = kappa + alpha*delta_kappa;
	tau     = tau + alpha*delta_tau;
	theta   = theta + alpha*delta_theta;
	iter	= iter + 1;
end

% �ŏI������
x_star      = x / tau;
y_star      = y / tau;
s_star      = y / tau;
kappa_star  = kappa;
tau_star    = tau;
isPrimalFeasible = true;
isDualFeasible   = true;



% �I������ƕ\��
if true
	if alpha < alphaMin
		 fprintf('�X�e�b�v�T�C�Y���������Ȃ肷���܂���\n');
		 fprintf('�Ō�̔����X��x=%f, y=%f, s=%f�ł�\n', x_star, y_star, s_star);
	else
		 fprintf('�������ȑo�Ό^�̖��̍œK���Ɏ������܂���\n');
		 fprintf('kappa = %f, tau = %f\n', kappa_star, tau_star);
		 if kappa > tau
			  fprintf('tau���������Ȃ�܂���\n');
			  if dot(b,y) > 0
					fprintf('���肪���s�s�\�ł�\n');
			  elseif dot(c,x) < 0
					fprintf('�o�Ζ�肪���s�s�\�ł�\n');
			  else
					fprintf('�N����Ȃ��P�[�X���N���Ă��܂�\n');
			  end
		 else % kappa <= tau�̏ꍇ
			  fprintf('kappa���������Ȃ�C�����̍œK����������܂���\n');
%			  fprintf('�œK����\n');
%			  fprintf('x = %f\n', x_star);
%			  fprintf('y = %f\n', y_star);
%			  fprintf('s = %f\n', s_star);
		 end
	end
end
end


function [d_bar] = solveSchur(A, G_bar, U_dense, V_dense, diag_x, diag_s_inv, a_p, a_d, a_u, denseIndex)
% Shur �⊮�s����������[�`��
% d_bar_y�̌v�Z
% ind :�u���x�N�g��
[m, ~] = size(A);
r = a_p - A * diag_s_inv * (a_u + diag_x*a_d);


%[zeta, ~] = pcg(G_bar, r, 1e-6, 20);
zeta = CGMethod(G_bar, r, 1e-3, 100);
if false
    G_U = zeros(m, denseIndex);
    for i = 1:denseIndex
        %[G_U(:,i), ~] = pcg(G_bar, U_dense(:, i), 1e-6, 20);
        G_U(:,i) = CGMethod(G_bar, U_dense(:, i), 1e-3, 20);
    end
end
if true
    G_U = CGMethodMat(G_bar, U_dense, 1e-0, 100, denseIndex);
end
%fprintf('fro = %f\n', norm((G_U_test - G_U), 'fro'));

I = speye(denseIndex);
d_bar_y = zeta - G_U * ((I + V_dense.'*G_U) \ (V_dense.'*zeta));

d_bar_s = -A.' * d_bar_y - a_d;
d_bar_x = diag_s_inv * (a_u - diag_x * d_bar_s);
d_bar = sparse([d_bar_x; d_bar_y; d_bar_s]);



%�����������Ă��邩�̊m�F
if false
	check_p = A * d_bar_x - a_p;
	check_d = -A.' * d_bar_y - d_bar_s - a_d;
	check_u = diag_s_inv^-1 * d_bar_x + diag_x * d_bar_s - a_u;
	fprintf('%f, %f, %f \n', norm(check_p), norm(check_d), norm(check_u));
end
end

function [x_opt, i] = CGMethod(A, b, eps, maxiter)
% implement of conjugate gradient method

%################################
%���P�_
%A��sparse��dense�̕����𕪂��� -> �s�ŕ����邱�ƂɂȂ�
%################################

[m, ~] = size(A);
x_old = sparse(m, 1);
x_new = sparse(m, 1);

r_old = b - A*x_old;
r_new = r_old;
p_old = r_old;
p_new = p_old;

for i = 1:maxiter
	% �x�N�g���̍X�V
	x_old = x_new; r_old = r_new; p_old = p_new;
	
    % ���̍s�ł��Ȃ�̎��Ԃ��������Ă���
	y = A*p_old;
	alpha = dot(r_old, r_old) / dot(p_old, y);
	x_new = x_old + alpha*p_old;
	r_new = r_old - alpha*y;
	
	% �I������
	if norm(r_new) < eps
		break;
	end
	beta = dot(r_new, r_new) / dot(r_old, r_old);
	p_new = r_new + beta*p_old;
end
%fprintf('iter = %d\n', i);
x_opt = x_new;
end


function [x_opt_mat] = CGMethodMat(A, b_mat, eps, maxiter, denseIndex)
% implement of conjugate gradient method with using matrix

%################################
%���P�_...
%�s��v�Z���ł���悤�ɂ��� -> r, p, x, �S�Ă��s��
%A��dense��sparse�𕪉����邱�Ƃō������H�H
%################################

[m, ~] = size(A);
% sparse�ɂ���K�v�����邩�킩��Ȃ�
x_old = zeros(m, denseIndex);
x_new = zeros(m, denseIndex);

r_old = b_mat - A*x_old;
r_new = r_old;
p_old = r_old;
p_new = p_old;

for i = 1:maxiter
	% �x�N�g���̍X�V
	x_old = x_new; r_old = r_new; p_old = p_new;
	
	y = A*p_old;
    alpha = sum((r_old .* r_old), 1) ./ sum((p_old .* y), 1);
	x_new = x_old + alpha.*p_old;
	r_new = r_old - alpha.*y;
	
	% �I������
    if norm(r_new, 'fro') < eps
		break;
    end
    beta = sum((r_new .* r_new), 1) ./ sum((r_old .* r_old), 1);
	p_new = r_new + beta.*p_old;
    
end
fprintf('iter_mat = %d\n', i);
x_opt_mat = x_new;

end


function [output_mat] = calculation(D, S_1, S_2, S_3)
% sparse �� dense �ɂ��ĕ����Čv�Z������
% [D, S_1; S_2, S_3] * [D, S_1; S_2, S_3]^T

tmp1 = D*S_2.';

out_1_1 = D*D.' + S_1*S_1.';
out_1_2 = D*S_2.' + S_1*S_3.';
out_2_1 = S_2*D.' + S_3*S_1.';
out_2_2 = S_2*S_2.' + S_3*S_3.';

output_mat = [out_1_1, out_1_2; out_2_1, out_2_2];
end
