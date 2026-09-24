# NonAuRA
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Core Exceptions
# ============================================================

class GuardrailViolation(RuntimeError):
    pass


class TransactionConflict(GuardrailViolation):
    pass


class FactConflict(GuardrailViolation):
    pass


class ClockSkewError(GuardrailViolation):
    pass


class HumanReviewRequired(GuardrailViolation):
    pass


# ============================================================
# Fair Async Lock
# ============================================================

class FairRWLock:
    """
    Non-reentrant by design.
    Public methods do not acquire other locks while already holding this lock.
    """

    def __init__(self, reader_batch: int = 32):
        self._condition = asyncio.Condition()
        self._active_readers = 0
        self._active_writer = False
        self._waiting_readers = 0
        self._waiting_writers = 0
        self._reader_batch = reader_batch
        self._reader_budget = reader_batch

    async def acquire_read(self) -> None:
        async with self._condition:
            self._waiting_readers += 1
            try:
                while True:
                    if self._active_writer:
                        await self._condition.wait()
                        continue
                    if self._waiting_writers == 0:
                        self._reader_budget = self._reader_batch
                        break
                    if self._reader_budget > 0:
                        self._reader_budget -= 1
                        break
                    await self._condition.wait()

                self._active_readers += 1
            finally:
                self._waiting_readers -= 1

    async def release_read(self) -> None:
        async with self._condition:
            self._active_readers -= 1
            if self._active_readers == 0:
                self._condition.notify_all()

    async def acquire_write(self) -> None:
        async with self._condition:
            self._waiting_writers += 1
            try:
                while self._active_writer or self._active_readers > 0:
                    await self._condition.wait()
                self._active_writer = True
            finally:
                self._waiting_writers -= 1

    async def release_write(self) -> None:
        async with self._condition:
            self._active_writer = False
            self._reader_budget = self._reader_batch
            self._condition.notify_all()


# ============================================================
# Fact Store
# ============================================================

@dataclass(frozen=True)
class Fact:
    key: str
    value: Any
    source: str
    confidence: float
    version: int
    created_at_epoch: float
    expires_at_epoch: Optional[float]

    def is_expired(self, now: float) -> bool:
        return (
            self.expires_at_epoch is not None
            and now >= self.expires_at_epoch
        )


@dataclass(frozen=True)
class PreparedFact:
    key: str
    value: Any
    source: str
    confidence: float
    created_at_epoch: float
    expires_at_epoch: Optional[float]


@dataclass(frozen=True)
class FactTransaction:
    observed_versions: Dict[str, int]
    updates: Dict[str, PreparedFact]


@dataclass(frozen=True)
class FactRequirement:
    key: str
    expected_value: Any = None
    min_confidence: float = 0.80
    max_age_seconds: Optional[float] = None
    allowed_sources: Optional[frozenset[str]] = None
    required: bool = True


class FactStore:
    def __init__(
        self,
        *,
        max_cleanup_batch: int = 256,
        cleanup_interval_seconds: float = 30.0,
        max_clock_skew_seconds: float = 30.0,
    ):
        self._facts: Dict[str, Fact] = {}
        self._versions: Dict[str, int] = {}
        self._version = 0
        self._lock = FairRWLock()

        self._max_cleanup_batch = max_cleanup_batch
        self._cleanup_interval_seconds = cleanup_interval_seconds
        self._max_clock_skew_seconds = max_clock_skew_seconds

        self._cleanup_queue: Deque[Tuple[str, int]] = deque()
        self._queued_cleanup: set[Tuple[str, int]] = set()

        self._sweeper_task: Optional[asyncio.Task] = None
        self._stop_sweeper = asyncio.Event()

    @staticmethod
    def now() -> float:
        return time.time()

    def _validate_clock(self, timestamp: float, *, now: Optional[float] = None) -> None:
        now = self.now() if now is None else now
        if abs(timestamp - now) > self._max_clock_skew_seconds:
            raise ClockSkewError(
                f"Fact timestamp exceeds clock skew window: timestamp={timestamp}, now={now}"
            )

    def _enqueue_cleanup_locked(self, key: str, version: int) -> None:
        entry = (key, version)
        if entry not in self._queued_cleanup:
            self._queued_cleanup.add(entry)
            self._cleanup_queue.append(entry)

    def _cleanup_locked(self, now: float) -> int:
        removed = 0
        scanned = 0

        while self._cleanup_queue and scanned < self._max_cleanup_batch:
            key, version = self._cleanup_queue.popleft()
            self._queued_cleanup.discard((key, version))
            scanned += 1

            fact = self._facts.get(key)
            if fact is None:
                continue
            if fact.version != version:
                continue

            if fact.is_expired(now):
                self._facts.pop(key, None)
                self._versions.pop(key, None)
                removed += 1
                continue

            if fact.expires_at_epoch is not None:
                self._enqueue_cleanup_locked(key, fact.version)

        return removed

    async def start_sweeper(self) -> None:
        if self._sweeper_task is None:
            self._stop_sweeper.clear()
            self._sweeper_task = asyncio.create_task(self._sweeper_loop())

    async def stop_sweeper(self) -> None:
        self._stop_sweeper.set()
        if self._sweeper_task is not None:
            await self._sweeper_task
            self._sweeper_task = None

    async def _sweeper_loop(self) -> None:
        while not self._stop_sweeper.is_set():
            try:
                await asyncio.wait_for(self._stop_sweeper.wait(), timeout=self._cleanup_interval_seconds)
            except asyncio.TimeoutError:
                await self.remove_expired()

    async def remove_expired(self) -> int:
        now = self.now()
        await self._lock.acquire_write()
        try:
            return self._cleanup_locked(now)
        finally:
            await self._lock.release_write()

    async def get(self, key: str) -> Optional[Fact]:
        now = self.now()
        await self._lock.acquire_read()
        try:
            fact = self._facts.get(key)
            if fact is None:
                return None
            if fact.is_expired(now):
                return None
            return fact
        finally:
            await self._lock.release_read()

    async def satisfies(self, requirement: FactRequirement) -> bool:
        requirement_key = requirement.key
        if not requirement_key:
            raise ValueError("Fact key required")
        if not 0.0 <= requirement.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if requirement.max_age_seconds is not None and requirement.max_age_seconds < 0:
            raise ValueError("max_age_seconds cannot be negative")

        await self._lock.acquire_read()
        try:
            fact = self._facts.get(requirement_key)
            if fact is None:
                return not requirement.required

            now = self.now()
            if fact.is_expired(now):
                return False
            if fact.confidence < requirement.min_confidence:
                return False
            if requirement.expected_value is not None and fact.value != requirement.expected_value:
                return False
            if requirement.max_age_seconds is not None:
                age = now - fact.created_at_epoch
                if age < -self._max_clock_skew_seconds:
                    return False
                if age > requirement.max_age_seconds:
                    return False
            if requirement.allowed_sources is not None and fact.source not in requirement.allowed_sources:
                return False
            return True
        finally:
            await self._lock.release_read()

    async def prepare(
        self,
        updates: Dict[str, Any],
        *,
        source: str,
        confidence: float = 0.90,
        ttl_seconds: Optional[float] = None,
    ) -> FactTransaction:
        if not source:
            raise FactConflict("Fact source is required")
        if not 0.0 <= confidence <= 1.0:
            raise FactConflict("Fact confidence must be between 0 and 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise FactConflict("ttl_seconds must be positive")

        now = self.now()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None

        observed_versions: Dict[str, int] = {}
        prepared: Dict[str, PreparedFact] = {}

        await self._lock.acquire_read()
        try:
            for key, value in updates.items():
                existing = self._facts.get(key)
                version = self._versions.get(key, 0)
                observed_versions[key] = version

                if existing is not None and existing.is_expired(now):
                    existing = None

                if existing is not None and existing.value != value:
                    raise FactConflict(f"Conflicting fact for key '{key}'")

                prepared_fact = PreparedFact(
                    key=key,
                    value=value,
                    source=source,
                    confidence=confidence,
                    created_at_epoch=now,
                    expires_at_epoch=expires_at,
                )

                self._validate_clock(prepared_fact.created_at_epoch, now=now)
                prepared[key] = prepared_fact

            return FactTransaction(observed_versions=observed_versions, updates=prepared)
        finally:
            await self._lock.release_read()

    async def commit(self, transaction: FactTransaction) -> None:
        now = self.now()

        await self._lock.acquire_write()
        try:
            # Preflight validation before mutation
            for prepared in transaction.updates.values():
                self._validate_clock(prepared.created_at_epoch, now=now)
                if not prepared.source:
                    raise FactConflict(f"Missing source for '{prepared.key}'")
                if not 0.0 <= prepared.confidence <= 1.0:
                    raise FactConflict(f"Invalid confidence for '{prepared.key}'")
                if prepared.expires_at_epoch is not None and prepared.expires_at_epoch <= prepared.created_at_epoch:
                    raise FactConflict(f"Invalid ttl for '{prepared.key}'")

            # Version validation
            for key, expected_version in transaction.observed_versions.items():
                actual = self._versions.get(key, 0)
                if actual != expected_version:
                    raise TransactionConflict(
                        f"Transaction conflict for '{key}': expected={expected_version}, actual={actual}"
                    )

            # Build replacement state in memory first
            next_facts = dict(self._facts)
            next_versions = dict(self._versions)
            next_version = self._version

            for key, prepared in transaction.updates.items():
                existing = next_facts.get(key)
                if existing is not None and not existing.is_expired(now) and existing.value != prepared.value:
                    raise FactConflict(f"Value changed before commit for '{key}'")

                next_version += 1
                next_versions[key] = next_version
                next_facts[key] = Fact(
                    key=prepared.key,
                    value=prepared.value,
                    source=prepared.source,
                    confidence=prepared.confidence,
                    version=next_version,
                    created_at_epoch=prepared.created_at_epoch,
                    expires_at_epoch=prepared.expires_at_epoch,
                )

            self._version = next_version
            self._facts = next_facts
            self._versions = next_versions

            for key, fact in next_facts.items():
                if fact.expires_at_epoch is not None:
                    self._enqueue_cleanup_locked(key, fact.version)

            self._cleanup_locked(now)

        finally:
            await self._lock.release_write()


# ============================================================
# Utilities
# ============================================================

_VOLATILE_KEYS = {
    "timestamp", "created_at", "updated_at", "request_id", "session_id",
    "trace_id", "nonce", "retry", "page", "offset", "limit", "count",
    "confidence"
}


def canonicalize(value: Any, *, visited: Optional[set[int]] = None, depth: int = 0, max_depth: int = 12, max_items: int = 256) -> Any:
    if visited is None:
        visited = set()
    if depth > max_depth:
        return "<max-depth>"

    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "<non-finite>"
        return round(value, 2)
    if isinstance(value, str):
        s = re.sub(r"\b(try again|retry now|once more|please retry)\b", "<retry>", value.lower().strip(), flags=re.IGNORECASE)
        return s[:2048]

    oid = id(value)
    if oid in visited:
        return "<cycle>"
    visited.add(oid)
    try:
        if isinstance(value, dict):
            out = {}
            for key, v in sorted(value.items(), key=lambda x: str(x[0]))[:max_items]:
                key_s = str(key)
                if key_s in _VOLATILE_KEYS:
                    continue
                out[key_s] = canonicalize(v, visited=visited, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            return out

        if isinstance(value, (list, tuple)):
            return [canonicalize(item, visited=visited, depth=depth + 1, max_depth=max_depth, max_items=max_items) for item in list(value)[:max_items]]

        if isinstance(value, (set, frozenset)):
            vals = [canonicalize(item, visited=visited, depth=depth + 1, max_depth=max_depth, max_items=max_items) for item in list(value)[:max_items]]
            return sorted(vals, key=lambda item: json.dumps(item, sort_keys=True, default=str))

        return {"__custom_type__": f"{type(value).__module__}.{type(value).__qualname__}"}
    finally:
        visited.remove(oid)


def stable_fingerprint(name: str, intent: Optional[str], args: Dict[str, Any]) -> str:
    payload = {
        "name": name,
        "intent": intent or name,
        "args": canonicalize(args),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LoopDetector:
    def __init__(self, max_steps: int = 50, history_window: int = 32, max_cycle_length: int = 12):
        self.max_steps = max_steps
        self.history_window = history_window
        self.max_cycle_length = max_cycle_length

    def check(self, history: Deque[str], current_fingerprint: str) -> None:
        if len(history) >= self.max_steps:
            raise GuardrailViolation(f"Maximum execution steps exceeded: {self.max_steps}")
        sequence = list(history)[-self.history_window:] + [current_fingerprint]
        if sequence.count(current_fingerprint) >= 3:
            raise GuardrailViolation("Repeated action fingerprint detected")
        max_period = min(self.max_cycle_length, len(sequence) // 2)
        for period in range(1, max_period + 1):
            if len(sequence) < period * 2:
                continue
            left = sequence[-period * 2:-period]
            right = sequence[-period:]
            if left == right:
                raise GuardrailViolation(f"Cycle detected with period {period}")


# ============================================================
# Agent Model
# ============================================================

class AgentDepartment(str, Enum):
    TRAVEL_TRANSPORT = "travel_transport"
    HOSPITALITY = "hospitality_accommodation"
    ENTERTAINMENT = "entertainment_events"
    TRANSACTION_FINANCE = "transaction_finance"
    CORE_ORCHESTRATION = "core_sync_orchestration"
    USER_CARE = "user_experience_care"


class AgentRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    department: AgentDepartment
    responsibility: str
    capabilities: Tuple[str, ...] = ()
    risk: AgentRisk = AgentRisk.MEDIUM
    read_facts: Tuple[str, ...] = ()
    write_facts: Tuple[str, ...] = ()
    requires_approval: bool = False
    max_concurrency: int = 4


# All 46 agents
AGENT_CATALOG: Tuple[AgentSpec, ...] = (
    AgentSpec("FlightFinderAgent", AgentDepartment.TRAVEL_TRANSPORT, "Aggregate airline routes, live prices and seat availability", ("flight_search","price_comparison","seat_availability"), AgentRisk.MEDIUM, ("trip.origin","trip.destination","trip.dates"), ("flight.options",), False, 4),
    AgentSpec("TrainSeatAgent", AgentDepartment.TRAVEL_TRANSPORT, "Check rail routes, berth availability and waitlist status", ("train_search","berth_check","waitlist_tracking"), AgentRisk.MEDIUM, ("trip.origin","trip.destination","trip.dates"), ("train.options",), False, 4),
    AgentSpec("BusRouteAgent", AgentDepartment.TRAVEL_TRANSPORT, "Track bus schedules, prices and seat maps", ("bus_search","schedule_lookup","seat_map"), AgentRisk.MEDIUM, ("trip.origin","trip.destination","trip.dates"), ("bus.options",), False, 4),
    AgentSpec("CabHailingAgent", AgentDepartment.TRAVEL_TRANSPORT, "Request last-mile rides from on-demand transport services", ("cab_quote","cab_booking","driver_tracking"), AgentRisk.HIGH, ("trip.pickup","trip.dropoff"), ("cab.booking",), True, 3),
    AgentSpec("LuggagePolicyAgent", AgentDepartment.TRAVEL_TRANSPORT, "Calculate baggage limits, dimensions and excess charges", ("baggage_rules","fee_calculation"), AgentRisk.LOW, ("flight.selected","traveler.profile"), ("luggage.policy",), False, 4),
    AgentSpec("RefundPolicyAgent", AgentDepartment.TRAVEL_TRANSPORT, "Calculate cancellation penalties and refund eligibility", ("refund_rules","penalty_calculation"), AgentRisk.HIGH, ("booking.selected","booking.created_at"), ("refund.policy",), False, 4),
    AgentSpec("RouteOptimizerAgent", AgentDepartment.TRAVEL_TRANSPORT, "Generate alternate multi-modal routes during disruptions", ("route_optimization","delay_rerouting"), AgentRisk.MEDIUM, ("flight.options","train.options","bus.options"), ("route.recommended",), False, 4),
    AgentSpec("TransitVisaAgent", AgentDepartment.TRAVEL_TRANSPORT, "Verify transit visa requirements from nationality and itinerary", ("visa_requirement_check","layover_analysis"), AgentRisk.HIGH, ("traveler.nationality","trip.itinerary"), ("visa.transit_requirement",), True, 3),
    AgentSpec("DelayPredictionAgent", AgentDepartment.TRAVEL_TRANSPORT, "Estimate delay probabilities for travel corridors", ("delay_prediction","risk_scoring"), AgentRisk.LOW, ("trip.itinerary","historical.transit_data"), ("delay.prediction",), False, 4),

    AgentSpec("HotelSearchAgent", AgentDepartment.HOSPITALITY, "Search hotels by constraints, budget and verified ratings", ("hotel_search","availability_check","rating_filter"), AgentRisk.MEDIUM, ("trip.destination","trip.dates","budget.limit"), ("hotel.options",), False, 4),
    AgentSpec("RoomConfigAgent", AgentDepartment.HOSPITALITY, "Configure rooms, beds and additional occupants", ("room_configuration","occupancy_validation"), AgentRisk.MEDIUM, ("hotel.selected","traveler.party"), ("room.configuration",), False, 4),
    AgentSpec("AmenitiesAgent", AgentDepartment.HOSPITALITY, "Verify property amenities and critical facilities", ("amenity_verification","facility_filtering"), AgentRisk.LOW, ("hotel.selected","traveler.preferences"), ("hotel.amenities",), False, 4),
    AgentSpec("CheckInManagerAgent", AgentDepartment.HOSPITALITY, "Request early check-in and late check-out variations", ("early_checkin","late_checkout","property_contact"), AgentRisk.MEDIUM, ("hotel.booking","traveler.arrival_time"), ("hotel.checkin_options",), False, 4),
    AgentSpec("HomestayAgent", AgentDepartment.HOSPITALITY, "Find compliant residential and homestay listings", ("homestay_search","local_compliance_check"), AgentRisk.MEDIUM, ("trip.destination","traveler.preferences"), ("homestay.options",), False, 4),
    AgentSpec("ReviewAnalyzerAgent", AgentDepartment.HOSPITALITY, "Analyze reviews and detect suspicious testimonials", ("sentiment_analysis","review_fraud_detection"), AgentRisk.LOW, ("hotel.options",), ("hotel.review_scores",), False, 4),
    AgentSpec("LocationScoutAgent", AgentDepartment.HOSPITALITY, "Measure distances, connectivity and location safety", ("distance_calculation","location_safety"), AgentRisk.MEDIUM, ("hotel.selected","trip.destinations"), ("hotel.location_score",), False, 4),
    AgentSpec("ConciergeAgent", AgentDepartment.HOSPITALITY, "Handle property micro-requests and shuttle arrangements", ("shuttle_request","room_request","property_messaging"), AgentRisk.MEDIUM, ("hotel.booking","traveler.preferences"), ("hotel.concierge_requests",), False, 4),

    AgentSpec("ConcertTicketAgent", AgentDepartment.ENTERTAINMENT, "Monitor and secure high-demand concert tickets", ("concert_search","ticket_hold","seat_selection"), AgentRisk.HIGH, ("event.preferences","budget.limit"), ("concert.options",), True, 3),
    AgentSpec("MovieShowAgent", AgentDepartment.ENTERTAINMENT, "Track cinema showtimes and seating preferences", ("cinema_search","showtime_lookup","seat_selection"), AgentRisk.LOW, ("trip.destination","event.preferences"), ("movie.options",), False, 4),
    AgentSpec("AmusementParkAgent", AgentDepartment.ENTERTAINMENT, "Find park tickets, fast-track queues and tours", ("park_ticket_search","fast_track","tour_packages"), AgentRisk.MEDIUM, ("trip.destination","trip.dates","budget.limit"), ("park.options",), False, 4),
    AgentSpec("LocalGuideAgent", AgentDepartment.ENTERTAINMENT, "Curate walking tours and historical attractions", ("tour_search","attraction_recommendation"), AgentRisk.LOW, ("traveler.interests","trip.destination"), ("tour.options",), False, 4),
    AgentSpec("RestaurantResAgent", AgentDepartment.ENTERTAINMENT, "Find and reserve restaurant tables", ("restaurant_search","table_reservation","waitlist"), AgentRisk.HIGH, ("trip.destination","traveler.dietary_needs"), ("restaurant.options",), True, 3),
    AgentSpec("EventAlertAgent", AgentDepartment.ENTERTAINMENT, "Monitor event cancellations and schedule changes", ("event_monitoring","change_alerts"), AgentRisk.LOW, ("event.bookings",), ("event.alerts",), False, 4),
    AgentSpec("NightlifeAgent", AgentDepartment.ENTERTAINMENT, "Find safe evening events and social meetups", ("nightlife_search","safety_filtering"), AgentRisk.MEDIUM, ("trip.destination","traveler.preferences"), ("nightlife.options",), False, 4),

    AgentSpec("PaymentGatewayAgent", AgentDepartment.TRANSACTION_FINANCE, "Execute payment handshakes through approved payment rails", ("payment_authorization","payment_capture"), AgentRisk.CRITICAL, ("payment.amount","payment.currency","payment.approval"), ("payment.status","payment.transaction_id"), True, 1),
    AgentSpec("FraudDetectionAgent", AgentDepartment.TRANSACTION_FINANCE, "Evaluate transaction signatures and velocity anomalies", ("fraud_scoring","velocity_check","risk_blocking"), AgentRisk.HIGH, ("payment.request","traveler.profile"), ("payment.fraud_score",), False, 4),
    AgentSpec("CurrencyConverterAgent", AgentDepartment.TRANSACTION_FINANCE, "Fetch FX rates and calculate cross-border settlement values", ("fx_rate","currency_conversion"), AgentRisk.MEDIUM, ("payment.amount","payment.currency"), ("payment.converted_amount",), False, 4),
    AgentSpec("DiscountCouponAgent", AgentDepartment.TRANSACTION_FINANCE, "Find and validate applicable discounts and loyalty points", ("coupon_search","coupon_validation","loyalty_points"), AgentRisk.MEDIUM, ("booking.cart","traveler.memberships"), ("payment.discount",), False, 4),
    AgentSpec("CorporateBillingAgent", AgentDepartment.TRANSACTION_FINANCE, "Generate regulatory invoices and apply corporate tax profiles", ("invoice_generation","tax_calculation","gst_validation"), AgentRisk.HIGH, ("payment.transaction_id","corporate.profile"), ("billing.invoice",), False, 4),
    AgentSpec("LedgerAgent", AgentDepartment.TRANSACTION_FINANCE, "Write immutable audit-ready transaction records", ("ledger_append","audit_recording"), AgentRisk.CRITICAL, ("payment.status","payment.transaction_id"), ("ledger.entry",), False, 1),
    AgentSpec("RefundTrackerAgent", AgentDepartment.TRANSACTION_FINANCE, "Poll payment rails until refunds are verified", ("refund_polling","refund_verification"), AgentRisk.HIGH, ("refund.request",), ("refund.status",), False, 4),
    AgentSpec("ExpenseSplitterAgent", AgentDepartment.TRANSACTION_FINANCE, "Split booking expenses across travelers or cost centers", ("expense_split","cost_center_allocation"), AgentRisk.MEDIUM, ("booking.cart","traveler.party"), ("expense.allocation",), False, 4),

    AgentSpec("FactStoreCoordinator", AgentDepartment.CORE_ORCHESTRATION, "Coordinate prepare and commit cycles across agents", ("fact_prepare","fact_commit","state_coordination"), AgentRisk.CRITICAL, (), (), False, 1),
    AgentSpec("LockManagerAgent", AgentDepartment.CORE_ORCHESTRATION, "Coordinate fair reader and writer access", ("lock_metrics","contention_monitoring"), AgentRisk.CRITICAL, (), (), False, 1),
    AgentSpec("ConflictResolverAgent", AgentDepartment.CORE_ORCHESTRATION, "Resolve transaction conflicts using fresh reads and bounded retries", ("conflict_retry","transaction_rebase","escalation"), AgentRisk.CRITICAL, (), (), False, 8),
    AgentSpec("ClockSynchronizerAgent", AgentDepartment.CORE_ORCHESTRATION, "Monitor clock drift and reject unsafe time-dependent state", ("clock_drift_check","time_health"), AgentRisk.HIGH, (), (), False, 4),
    AgentSpec("TTLSweeperAgent", AgentDepartment.CORE_ORCHESTRATION, "Evict expired facts and temporary state", ("ttl_cleanup","expired_fact_eviction"), AgentRisk.MEDIUM, (), (), False, 1),
    AgentSpec("DependencyAgent", AgentDepartment.CORE_ORCHESTRATION, "Enforce action dependencies and workflow ordering", ("dag_validation","dependency_resolution"), AgentRisk.CRITICAL, (), (), False, 1),
    AgentSpec("LoadBalancerAgent", AgentDepartment.CORE_ORCHESTRATION, "Distribute workloads across agent workers", ("load_balancing","capacity_routing"), AgentRisk.HIGH, (), (), False, 4),

    AgentSpec("PreferenceAgent", AgentDepartment.USER_CARE, "Store and retrieve user travel preferences", ("preference_storage","preference_matching"), AgentRisk.MEDIUM, ("traveler.profile",), ("traveler.preferences",), False, 4),
    AgentSpec("NotificationAgent", AgentDepartment.USER_CARE, "Send confirmations, alerts and receipts", ("email","sms","push_notification"), AgentRisk.MEDIUM, ("booking.status","payment.status"), ("notification.status",), False, 4),
    AgentSpec("BudgetEnforcerAgent", AgentDepartment.USER_CARE, "Enforce cumulative spending limits", ("budget_check","spend_tracking","budget_blocking"), AgentRisk.CRITICAL, ("budget.limit","payment.amount"), ("budget.remaining",), False, 1),
    AgentSpec("WeatherWatchAgent", AgentDepartment.USER_CARE, "Monitor weather threats affecting the itinerary", ("weather_monitoring","weather_alerts"), AgentRisk.MEDIUM, ("trip.destinations","trip.dates"), ("weather.alerts",), False, 4),
    AgentSpec("EmergencyAgent", AgentDepartment.USER_CARE, "Trigger emergency cancellation and safety workflows", ("emergency_stop","rapid_cancellation","safety_protocol"), AgentRisk.CRITICAL, ("emergency.alert","trip.bookings"), ("emergency.status",), True, 1),
    AgentSpec("VisaStatusAgent", AgentDepartment.USER_CARE, "Monitor pending international visa applications", ("visa_status","consular_api"), AgentRisk.HIGH, ("traveler.nationality","visa.application"), ("visa.status",), False, 4),
    AgentSpec("HealthSafetyAgent", AgentDepartment.USER_CARE, "Monitor health notices, quarantine and vaccine rules", ("health_advisory","quarantine_rules","vaccine_rules"), AgentRisk.HIGH, ("trip.destinations","trip.dates"), ("health.alerts",), False, 4),
)

# ============================================================
# Agent registry + concurrency
# ============================================================

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, AgentSpec] = {}
        self._handlers: Dict[str, Callable[[Any, AgentSpec], Awaitable[Any]]] = {}
        for spec in AGENT_CATALOG:
            self._agents[spec.name] = spec

    def get(self, name: str) -> AgentSpec:
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        return self._agents[name]

    def all(self) -> Tuple[AgentSpec, ...]:
        return tuple(self._agents.values())

    def names(self) -> Tuple[str, ...]:
        return tuple(self._agents.keys())

    def by_department(self, department: AgentDepartment) -> Tuple[AgentSpec, ...]:
        return tuple(spec for spec in self._agents.values() if spec.department == department)

    def register_handler(self, name: str, handler: Callable[[Any, AgentSpec], Awaitable[Any]]) -> None:
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        self._handlers[name] = handler

    def handler_for(self, name: str) -> Callable[[Any, AgentSpec], Awaitable[Any]]:
        if name not in self._handlers:
            raise GuardrailViolation(f"No handler registered for {name}")
        return self._handlers[name]


class AgentConcurrency:
    def __init__(self, registry: AgentRegistry):
        self._semaphores = {spec.name: asyncio.Semaphore(spec.max_concurrency) for spec in registry.all()}

    def get(self, name: str) -> asyncio.Semaphore:
        return self._semaphores[name]


# ============================================================
# Request/response
# ============================================================

@dataclass
class SwarmRequest:
    request_id: str
    user_id: str
    intent: str
    payload: Dict[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = False
    budget_limit: Optional[float] = None


@dataclass
class SwarmResponse:
    agent: str
    request_id: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    escalated: bool = False


# ============================================================
# Runtime
# ============================================================

class Action:
    def __init__(
        self,
        *,
        action_id: str,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        depends_on: Tuple[str, ...] = (),
        requires_facts: Tuple[FactRequirement, ...] = (),
        produces_facts: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        risk: float = 0.0,
        fact_ttl_seconds: Optional[float] = None,
    ):
        self.action_id = action_id
        self.name = name
        self.args = args or {}
        self.intent = intent
        self.depends_on = depends_on
        self.requires_facts = requires_facts
        self.produces_facts = produces_facts or {}
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.risk = risk
        self.fact_ttl_seconds = fact_ttl_seconds


class DAG:
    def __init__(self):
        self.nodes: Dict[str, Action] = {}

    def add(self, action: Action) -> None:
        if action.action_id in self.nodes:
            raise ValueError(f"Duplicate action id: {action.action_id}")
        self.nodes[action.action_id] = action
        self.validate()

    def validate(self) -> None:
        visited: set[str] = set()
        active: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in active:
                raise GuardrailViolation(f"DAG cycle detected at '{node_id}'")
            if node_id in visited:
                return
            if node_id not in self.nodes:
                raise GuardrailViolation(f"Unknown dependency '{node_id}'")
            active.add(node_id)
            for dep in self.nodes[node_id].depends_on:
                visit(dep)
            active.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    def ready(self, completed: set[str], running: set[str]) -> List[Action]:
        ready: List[Action] = []
        for action in self.nodes.values():
            if action.action_id in completed or action.action_id in running:
                continue
            if all(dep in completed for dep in action.depends_on):
                ready.append(action)
        return ready


async def invoke_tool(tool: Callable[..., Any], args: Dict[str, Any], timeout: float) -> Any:
    async def call() -> Any:
        if asyncio.iscoroutinefunction(tool):
            return await tool(**args)
        return await asyncio.to_thread(tool, **args)
    return await asyncio.wait_for(call(), timeout=timeout)


class AgentRuntime:
    def __init__(self, *, task_id: str, facts: FactStore, tools: Dict[str, Callable], max_failures: int = 3):
        self.task_id = task_id
        self.facts = facts
        self.tools = tools
        self.max_failures = max_failures
        self.failures = 0
        self.history: Deque[str] = deque(maxlen=50)
        self.loop_detector = LoopDetector()
        self._state_lock = asyncio.Lock()

    async def validate_action(self, action: Action) -> None:
        for requirement in action.requires_facts:
            ok = await self.facts.satisfies(requirement)
            if not ok:
                raise GuardrailViolation(f"Fact requirement failed: {requirement.key}")
        if action.risk >= 0.8 and not action.requires_facts:
            raise GuardrailViolation(f"High-risk action '{action.action_id}' requires explicit fact requirements")

    async def _reserve_action(self, action: Action) -> None:
        async with self._state_lock:
            fp = stable_fingerprint(action.name, action.intent, action.args)
            self.loop_detector.check(self.history, fp)
            self.history.append(fp)

    async def run_action(self, action: Action) -> Any:
        await self._reserve_action(action)
        await self.validate_action(action)

        tool = self.tools.get(action.name)
        if tool is None:
            raise GuardrailViolation(f"Unknown tool: {action.name}")

        last_error = None
        for attempt in range(1, action.max_retries + 2):
            try:
                result = await invoke_tool(tool, action.args, action.timeout_seconds)
                if action.produces_facts:
                    tx = await self.facts.prepare(action.produces_facts, source=f"action:{action.action_id}", confidence=0.90, ttl_seconds=action.fact_ttl_seconds)
                    await self.facts.commit(tx)
                return result
            except Exception as exc:
                last_error = exc
                if attempt > action.max_retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 8.0))

        async with self._state_lock:
            self.failures += 1
            if self.failures >= self.max_failures:
                raise HumanReviewRequired(f"Failure budget exceeded for task {self.task_id}")

        raise last_error or RuntimeError(f"Action '{action.action_id}' failed")


# ============================================================
# Conflict Resolver
# ============================================================

class ConflictResolverAgent:
    def __init__(self, *, max_retries: int = 3, base_delay: float = 0.2):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def run(self, operation: Callable[[], Awaitable[Any]], *, high_risk: bool = False) -> Any:
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return await operation()
            except TransactionConflict as exc:
                last_error = exc
                if high_risk and attempt >= 2:
                    raise HumanReviewRequired("High-risk transaction conflict requires human review") from exc
                if attempt > self.max_retries:
                    break
                delay = min(self.base_delay * (2 ** (attempt - 1)), 5.0)
                delay += random.uniform(0.0, delay * 0.25)
                await asyncio.sleep(delay)
        if last_error is not None:
            raise HumanReviewRequired("Conflict retry budget exhausted") from last_error
        raise RuntimeError("Conflict retry failed")


# ============================================================
# Swarm runtime
# ============================================================

class AutonomousSwarm:
    def __init__(self, *, registry: AgentRegistry, facts: FactStore, conflict_resolver: Optional[ConflictResolverAgent] = None):
        self.registry = registry
        self.facts = facts
        self.conflict_resolver = conflict_resolver or ConflictResolverAgent()
        self._semaphores = {spec.name: asyncio.Semaphore(spec.max_concurrency) for spec in registry.all()}

    async def execute_agent(self, agent_name: str, request: SwarmRequest) -> SwarmResponse:
        spec = self.registry.get(agent_name)

        if spec.requires_approval and request.requires_human_approval is False:
            return SwarmResponse(agent=agent_name, request_id=request.request_id, ok=False, error=f"{agent_name} requires explicit approval", escalated=True)

        async def operation() -> Any:
            handler = self.registry.handler_for(agent_name)
            return await handler(request, spec)

        async with self._semaphores[agent_name]:
            try:
                output = await self.conflict_resolver.run(operation, high_risk=spec.risk in {AgentRisk.HIGH, AgentRisk.CRITICAL})
                return SwarmResponse(agent=agent_name, request_id=request.request_id, ok=True, output=output)
            except HumanReviewRequired as e:
                return SwarmResponse(agent=agent_name, request_id=request.request_id, ok=False, error=str(e), escalated=True)
            except Exception as e:
                return SwarmResponse(agent=agent_name, request_id=request.request_id, ok=False, error=str(e))

    async def execute_stage(self, agent_names: Iterable[str], request: SwarmRequest) -> List[SwarmResponse]:
        return await asyncio.gather(*(self.execute_agent(a, request) for a in agent_names))

    async def execute_plan(self, stages: List[List[str]], request: SwarmRequest) -> List[SwarmResponse]:
        results: List[SwarmResponse] = []
        for stage in stages:
            stage_results = await self.execute_stage(stage, request)
            results.extend(stage_results)
            failed = [r for r in stage_results if not r.ok]
            if failed:
                raise GuardrailViolation("Swarm stage failed: " + ", ".join(r.agent for r in failed))
        return results


# ============================================================
# Default handlers (safe, live-ready stubs)
# ============================================================

def make_default_handler(facts: FactStore):
    async def handler(request: SwarmRequest, spec: AgentSpec) -> Dict[str, Any]:
        await asyncio.sleep(0.02)

        # seed universal facts on the first run
        payload = request.payload or {}
        user_profile = payload.get("traveler", {})
        trip = payload.get("trip", {})
        budget_limit = request.budget_limit or payload.get("budget_limit")

        await facts.prepare(
            {
                "traveler.profile": user_profile,
                "trip.destination": trip.get("destination"),
                "budget.limit": budget_limit,
                "request.intent": request.intent,
            },
            source=f"agent:{spec.name}",
            confidence=0.88,
            ttl_seconds=300,
        )

        return {
            "agent": spec.name,
            "department": spec.department.value,
            "responsibility": spec.responsibility,
            "status": "planned",
            "request_id": request.request_id,
            "intent": request.intent,
            "capabilities": spec.capabilities,
            "decision_context": {
                "budget_limit": budget_limit,
                "trip": trip,
                "traveler": user_profile,
            },
        }

    return handler


# ============================================================
# Bootstrap
# ============================================================

async def build_swarm() -> Tuple[FactStore, AgentRegistry, AutonomousSwarm]:
    facts = FactStore(max_cleanup_batch=256, cleanup_interval_seconds=15.0, max_clock_skew_seconds=30.0)
    await facts.start_sweeper()

    registry = AgentRegistry()
    default_handler = make_default_handler(facts)

    for spec in registry.all():
        registry.register_handler(spec.name, default_handler)

    swarm = AutonomousSwarm(registry=registry, facts=facts, conflict_resolver=ConflictResolverAgent(max_retries=3))

    return facts, registry, swarm


# ============================================================
# Example
# ============================================================

async def run_demo() -> None:
    facts, registry, swarm = await build_swarm()

    request = SwarmRequest(
        request_id="req-001",
        user_id="u-100",
        intent="plan_trip",
        payload={
            "trip": {
                "origin": "DEL",
                "destination": "LHR",
                "dates": ["2026-10-10", "2026-10-20"],
            },
            "traveler": {
                "nationality": "IN",
                "dietary_needs": ["vegetarian"],
                "preferences": ["window-seat", "quiet-room"],
            },
            "event_preferences": {"concert": True, "restaurant": True},
            "budget_limit": 2800.0,
        },
        requires_human_approval=False,
        budget_limit=2800.0,
    )

    plan = [
        ["PreferenceAgent", "WeatherWatchAgent", "HealthSafetyAgent", "VisaStatusAgent"],
        ["FlightFinderAgent", "TrainSeatAgent", "BusRouteAgent", "HotelSearchAgent", "HomestayAgent"],
        ["RouteOptimizerAgent", "RoomConfigAgent", "AmenitiesAgent", "ReviewAnalyzerAgent", "LocationScoutAgent"],
        ["CurrencyConverterAgent", "DiscountCouponAgent", "FraudDetectionAgent", "BudgetEnforcerAgent"],
        ["PaymentGatewayAgent", "LedgerAgent", "NotificationAgent"],
    ]

    try:
        results = await swarm.execute_plan(plan, request)
        for item in results:
            print(json.dumps({"agent": item.agent, "ok": item.ok, "result": item.output, "error": item.error}, default=str, indent=2))
    finally:
        await facts.stop_sweeper()


if __name__ == "__main__":
    asyncio.run(run_demo())
