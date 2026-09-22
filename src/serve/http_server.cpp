#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
    {
        std::lock_guard lock(last_throughput_mutex_);
        last_throughput_        = report;
        last_throughput_at_     = std::chrono::steady_clock::now();
    }
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                        "application/json");
    });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        handle_metrics(res);
    });
    server_.Get("/slots", [this](const httplib::Request&, httplib::Response& res) {
        handle_slots(res);
    });
    server_.Get("/usage", [this](const httplib::Request&, httplib::Response& res) {
        handle_usage(res);
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

void HttpServer::handle_metrics(httplib::Response& res) const {
    using Clock  = std::chrono::steady_clock;
    static const Clock::time_point process_start = Clock::now();
    struct CounterDef {
        const char* name;
        const char* help;
        std::uint64_t value;
    };
    std::string out;
    auto emit_u64 = [&](const char* name, const char* help, std::uint64_t value,
                        const char* labels = "") {
        out += std::string("# HELP ") + name + ' ' + help + "\n# TYPE " + name + " counter\n" +
               name + labels + ' ' + std::to_string(value) + "\n";
    };
    auto emit_gauge = [&](const char* name, const char* help, std::uint64_t value) {
        out += std::string("# HELP ") + name + ' ' + help + "\n# TYPE " + name + " gauge\n" +
               name + ' ' + std::to_string(value) + "\n";
    };
    auto emit_dbl = [&](const char* name, const char* help, double value) {
        out += std::string("# HELP ") + name + ' ' + help + "\n# TYPE " + name + " gauge\n" +
               name + ' ' + std::to_string(value) + "\n";
    };

    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(out, "text/plain; version=0.0.4");
        return;
    }
    const ninfer::RuntimeStats stats = service_->runtime_stats();
    const ninfer::LoadSummary load   = service_->load_summary();
    // Fresh 5s-window rates from the stats reporter (0 when idle — lifetime averages
    // over full uptime are meaningless on a bursty server).
    double window_prefill_tps = 0.0;
    double window_decode_tps  = 0.0;
    bool   window_fresh       = false;
    {
        std::lock_guard lock(last_throughput_mutex_);
        const double age_s = std::chrono::duration<double>(
                                 Clock::now() - last_throughput_at_).count();
        window_fresh = age_s < 10.0 && last_throughput_.interval_seconds > 0.0;
        if (window_fresh) {
            window_prefill_tps = static_cast<double>(last_throughput_.computed_prefill_tokens) /
                                 last_throughput_.interval_seconds;
            window_decode_tps = static_cast<double>(last_throughput_.committed_decode_tokens) /
                                last_throughput_.interval_seconds;
        }
    }

    emit_u64("llamacpp:prompt_tokens_total",
             "Total prompt tokens computed by prefill (cache hits excluded).",
             stats.computed_prefill_tokens);
    emit_u64("llamacpp:prompt_seconds_total", "Total wall seconds spent computing prefill.",
             0);
    emit_u64("llamacpp:tokens_predicted_total",
             "Total output tokens committed by decode.", stats.committed_decode_tokens);
    emit_u64("llamacpp:tokens_predicted_seconds_total",
             "Total wall seconds spent decoding.", 0);
    emit_gauge("llamacpp:requests_processing", "Requests currently running or prefilling.",
               static_cast<std::uint64_t>(stats.running_requests));
    emit_gauge("llamacpp:prompt_tokens_seconds", "Recent-window prefill throughput (tok/s).",
               window_prefill_tps);
    emit_gauge("llamacpp:predicted_tokens_seconds", "Recent-window decode throughput (tok/s).",
               window_decode_tps);
    emit_gauge("ninfer:throughput_window_fresh", "1 if the throughput window saw activity.",
               window_fresh ? 1 : 0);
    emit_gauge("llamacpp:requests_deferred", "Requests waiting in the admission queue.",
               static_cast<std::uint64_t>(stats.waiting_requests));
    emit_gauge("llamacpp:prompt_cache_tokens_total",
               "Prompt tokens served from checkpoint reuse (all reuse paths).",
               stats.reused_prompt_tokens);

    emit_u64("ninfer:requests_total", "Total generation requests admitted.",
             stats.root_selections + stats.private_endpoint_selections +
                 stats.private_turn_closure_selections +
                 stats.private_response_replay_selections +
                 stats.private_long_anchor_selections +
                 stats.shared_stable_prefix_selections);
    emit_u64("ninfer:decode_rounds_total", "Decode batch executions.", stats.decode_rounds);
    emit_u64("ninfer:decode_row_rounds_total",
             "Sum of per-row decode executions across batches.", stats.decode_row_rounds);
    emit_u64("ninfer:prefill_units_total", "Prefill compact-batch units executed.",
             stats.host_work.prefill_units);
    emit_u64("ninfer:prompt_cache_root_selections_total",
             "Requests whose reuse candidate was the shared root (cold prefill).",
             stats.root_selections);
    emit_u64("ninfer:prompt_cache_private_endpoint_selections_total",
             "Requests restored from their session endpoint (pure appends).",
             stats.private_endpoint_selections);
    emit_u64("ninfer:prompt_cache_turn_closure_selections_total",
             "Requests restored from a turn-closure checkpoint.",
             stats.private_turn_closure_selections);
    emit_u64("ninfer:prompt_cache_response_replay_selections_total",
             "Requests restored by response replay.", stats.private_response_replay_selections);
    emit_u64("ninfer:prompt_cache_long_anchor_selections_total",
             "Requests restored from a private long anchor (history rewrite).",
             stats.private_long_anchor_selections);
    emit_u64("ninfer:prompt_cache_shared_prefix_selections_total",
             "Requests reading a published shared prefix.",
             stats.shared_stable_prefix_selections);
    emit_gauge("ninfer:prefilling_requests", "Requests currently computing prefill.",
               stats.prefilling_requests);
    emit_gauge("ninfer:decode_ready_requests", "Requests holding a decode lane.",
               stats.decode_ready_requests);
    emit_gauge("ninfer:materializing_requests", "Requests restoring checkpoints.",
               stats.materializing_requests);
    emit_gauge("ninfer:capture_pending_requests", "Requests waiting on checkpoint capture.",
               stats.capture_pending_requests);
    emit_gauge("ninfer:device_state_occupied_slots", "Occupied device state slots.",
               stats.device_state_occupied_slots);
    emit_gauge("ninfer:host_state_occupied_slots", "Occupied host state slots.",
               stats.host_state_occupied_slots);
    emit_gauge("ninfer:device_main_kv_occupied_pages", "Occupied main KV pages on device.",
               stats.device_main_kv_occupied_pages);
    emit_u64("ninfer:active_captures_completed_total",
             "Checkpoint captures completed.", stats.active_captures_completed);
    emit_u64("ninfer:active_captures_aborted_total",
             "Checkpoint captures aborted.", stats.active_captures_aborted);
    emit_u64("ninfer:pressure_spill_pages_total", "KV pages spilled under pressure.",
             stats.pressure_spill_pages);
    emit_u64("ninfer:pressure_private_owners_evicted_total",
             "Private continuations evicted under pressure.",
             stats.pressure_private_owners_evicted);
    emit_dbl("ninfer:process_uptime_seconds", "Seconds since the stats endpoint was initialized.",
             std::chrono::duration<double>(Clock::now() - process_start).count());

    res.set_header("Cache-Control", "no-store");
    res.set_content(out, "text/plain; version=0.0.4");
}

void HttpServer::handle_slots(httplib::Response& res) const {
    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(R"({"error":"service not attached"})", "application/json");
        return;
    }
    const ninfer::RuntimeStats stats = service_->runtime_stats();
    const std::uint32_t max_lanes    = service_->engine_options().max_concurrency;
    nlohmann::json j                 = nlohmann::json::array();
    for (std::uint32_t lane = 0; lane < max_lanes; ++lane) {
        j.push_back(nlohmann::json{
            {"id", lane},
            {"busy", lane < stats.running_requests},
        });
    }
    nlohmann::json out{
        {"model", public_model_id_},
        {"max_concurrency", max_lanes},
        {"requests_processing", stats.running_requests},
        {"requests_waiting", stats.waiting_requests},
        {"prefilling", stats.prefilling_requests},
        {"decode_ready", stats.decode_ready_requests},
        {"materializing", stats.materializing_requests},
        {"device_state_slots", {{"occupied", stats.device_state_occupied_slots}}},
        {"host_state_slots", {{"occupied", stats.host_state_occupied_slots}}},
        {"kv_pages",
         {{"main_occupied", stats.device_main_kv_occupied_pages},
          {"backend_occupied", stats.device_backend_kv_occupied_pages}}},
        {"slots", std::move(j)},
    };
    res.set_content(out.dump(), "application/json");
}

void HttpServer::handle_usage(httplib::Response& res) const {
    using Clock  = std::chrono::steady_clock;
    static const Clock::time_point process_start = Clock::now();
    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(R"({"error":"service not attached"})", "application/json");
        return;
    }
    const ninfer::RuntimeStats stats = service_->runtime_stats();
    const double uptime_s =
        std::chrono::duration<double>(Clock::now() - process_start).count();

    const std::uint64_t total_prompt = stats.computed_prefill_tokens +
                                       stats.reused_prompt_tokens;
    const std::uint64_t cache_hits   = stats.reused_prompt_tokens;
    const double hit_rate = total_prompt ? 100.0 * cache_hits / total_prompt : 0.0;
    const double cache_efficiency =
        (stats.root_selections + stats.private_endpoint_selections +
         stats.private_turn_closure_selections + stats.private_response_replay_selections +
         stats.private_long_anchor_selections + stats.shared_stable_prefix_selections) != 0
            ? 100.0 * (stats.private_endpoint_selections +
                       stats.private_turn_closure_selections +
                       stats.private_response_replay_selections +
                       stats.private_long_anchor_selections +
                       stats.shared_stable_prefix_selections) /
                  (stats.root_selections + stats.private_endpoint_selections +
                   stats.private_turn_closure_selections +
                   stats.private_response_replay_selections +
                   stats.private_long_anchor_selections +
                   stats.shared_stable_prefix_selections)
            : 0.0;
    // Lifetime averages over full uptime are meaningless on a bursty server (idle time
    // dominates). Report the most recent 5s stats-reporter window instead: prefill and
    // decode tok/s while serving, with an activity flag so consumers can tell idle.
    double window_prefill_tps = 0.0;
    double window_decode_tps  = 0.0;
    std::uint64_t window_prefill_tokens = 0;
    std::uint64_t window_decode_tokens  = 0;
    double window_seconds               = 0.0;
    bool   window_active                = false;
    {
        std::lock_guard lock(last_throughput_mutex_);
        const double age_s = std::chrono::duration<double>(
                                 Clock::now() - last_throughput_at_).count();
        window_active        = age_s < 10.0;
        window_seconds       = last_throughput_.interval_seconds;
        window_prefill_tokens = last_throughput_.computed_prefill_tokens;
        window_decode_tokens  = last_throughput_.committed_decode_tokens;
        if (window_seconds > 0.0 && window_active) {
            window_prefill_tps = static_cast<double>(window_prefill_tokens) / window_seconds;
            window_decode_tps  = static_cast<double>(window_decode_tokens) / window_seconds;
        }
    }
    const int hours   = static_cast<int>(uptime_s / 3600);
    const int minutes = static_cast<int>(uptime_s / 60) % 60;

    nlohmann::json out{
        {"model", public_model_id_},
        {"uptime", {{"seconds", uptime_s},
                    {"human", std::to_string(hours) + "h " + std::to_string(minutes) + "m"}}},
        {"tokens",
         {{"input",
           {{"total", total_prompt},
            {"cache_hits", cache_hits},
            {"cache_computed", stats.computed_prefill_tokens},
            {"cache_hit_rate_pct", hit_rate}}},
          {"output", {{"total", stats.committed_decode_tokens}}},
          {"total", total_prompt + stats.committed_decode_tokens}}},
        {"throughput",
         {{"window_seconds", window_seconds},
          {"window_active", window_active},
          {"prefill_tok_per_s", window_prefill_tps},
          {"decode_tok_per_s", window_decode_tps},
          {"prefill_tokens_last_window", window_prefill_tokens},
          {"decode_tokens_last_window", window_decode_tokens}}},
        {"requests",
         {{"total", stats.root_selections + stats.private_endpoint_selections +
                        stats.private_turn_closure_selections +
                        stats.private_response_replay_selections +
                        stats.private_long_anchor_selections +
                        stats.shared_stable_prefix_selections},
          {"cache_miss_cold_starts", stats.root_selections},
          {"cache_hit_endpoint", stats.private_endpoint_selections},
          {"cache_hit_turn_closure", stats.private_turn_closure_selections},
          {"cache_hit_response_replay", stats.private_response_replay_selections},
          {"cache_hit_long_anchor", stats.private_long_anchor_selections},
          {"cache_hit_shared_prefix", stats.shared_stable_prefix_selections},
          {"cache_efficiency_pct", cache_efficiency}}},
        {"lanes",
         {{"capacity", service_->engine_options().max_concurrency},
          {"processing", stats.running_requests},
          {"waiting", stats.waiting_requests},
          {"decoding", stats.decode_ready_requests},
          {"prefilling", stats.prefilling_requests},
          {"restoring_checkpoints", stats.materializing_requests}}},
        {"health",
         {{"checkpoint_captures_completed", stats.active_captures_completed},
          {"checkpoint_captures_aborted", stats.active_captures_aborted},
          {"kv_pressure_spills", stats.pressure_spill_pages},
          {"sessions_evicted", stats.pressure_private_owners_evicted},
          {"kv_pages_occupied", stats.device_main_kv_occupied_pages}}},
    };
    res.set_header("Cache-Control", "no-store");
    res.set_content(out.dump(2), "application/json");
}



bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_name);
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
