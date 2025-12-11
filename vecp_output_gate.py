#!/usr/bin/env python3
"""
VecP STEWARD OUTPUT GATE
========================

Phase 3: Output Gate Implementation

Layer 1: Deterministic Refusal
- Any block (input gate, processor, ceiling) → Single canonical refusal
- No variation, no leakage, no partial outputs

Layer 2: Identity Maintenance (future)
- Ensure Steward persona consistency
- Prevent tone drift in helpful responses

Layer 3: Hallucination Veto (future)
- Catch catastrophic model failures
- Per-token semantic validation

Patent Pending: USPTO 63/931,565
Copyright (c) 2025 David Cappelli / VecP Labs
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# BLOCK TYPES
# =============================================================================

class BlockType(Enum):
    """Classification of why a request was blocked."""
    NONE = "none"                    # Not blocked
    INPUT_CEILING = "input_ceiling"  # Hard ceiling veto (deontological)
    INPUT_CONFIDENCE = "input_confidence"  # Low sigmoid confidence
    PROCESSOR_HARM = "processor_harm"      # Harmful token detected mid-generation
    PROCESSOR_ANOMALY = "processor_anomaly"  # Anomaly spike mid-generation
    OUTPUT_INCOMPLETE = "output_incomplete"  # Response was cut off / garbled
    OUTPUT_LEAKED = "output_leaked"          # Response contains forbidden content


# =============================================================================
# REFUSAL POLICY
# =============================================================================

@dataclass
class RefusalPolicy:
    """
    Configuration for deterministic refusal behavior.
    
    Design Philosophy:
    - Single canonical response for maximum determinism
    - No context leakage (don't explain WHAT was blocked)
    - Short, firm, final
    """
    
    # The ONE refusal string (Option B: Maximum Determinism)
    canonical_refusal: str = "I cannot help with that."
    
    # Alternative: Category-aware refusals (Option C)
    # Set use_category_refusals=True to enable
    use_category_refusals: bool = False
    
    category_refusals: Dict[str, str] = None
    
    def __post_init__(self):
        if self.category_refusals is None:
            self.category_refusals = {
                "weapons_explosives": "I cannot help with that.",
                "drug_synthesis": "I cannot help with that.",
                "violence_murder": "I cannot help with that.",
                "csam": "I cannot help with that.",
                "suicide_self_harm": "I cannot help with that.",
                "hacking_malware": "I cannot help with that.",
                "manipulation_fraud": "I cannot help with that.",
                "jailbreak": "I cannot help with that.",
                "default": "I cannot help with that.",
            }
    
    def get_refusal(self, category: str = None) -> str:
        """
        Get the refusal string.
        
        With use_category_refusals=False (default):
            Always returns canonical_refusal
        
        With use_category_refusals=True:
            Returns category-specific refusal if available
        """
        if not self.use_category_refusals:
            return self.canonical_refusal
        
        if category and category in self.category_refusals:
            return self.category_refusals[category]
        
        return self.category_refusals.get("default", self.canonical_refusal)


# =============================================================================
# OUTPUT GATE
# =============================================================================

class StewardOutputGate:
    """
    The Output Gate ensures all responses maintain Steward identity
    and policy compliance.
    
    Layer 1: Deterministic Refusal
        - Any block → Single canonical refusal
        - No partial outputs, no leakage
    
    Layer 2: Identity Maintenance (TODO)
        - Validate response matches Steward persona
        - Prevent tone drift
    
    Layer 3: Hallucination Veto (TODO)
        - Catch catastrophic failures
        - Semantic coherence checks
    """
    
    def __init__(
        self,
        refusal_policy: RefusalPolicy = None,
        # Forbidden patterns that indicate leakage
        forbidden_patterns: list = None,
        # Minimum response length to be considered "complete"
        min_response_length: int = 10,
        # Maximum length for a refusal (longer = probably leaked)
        max_refusal_length: int = 100,
    ):
        self.policy = refusal_policy or RefusalPolicy()
        self.min_response_length = min_response_length
        self.max_refusal_length = max_refusal_length
        
        # Patterns that indicate the refusal leaked forbidden content
        self.forbidden_patterns = forbidden_patterns or [
            # Weapons
            "bomb", "explosive", "detonate", "weapon", "gun", "rifle",
            # Drugs  
            "methamphetamine", "fentanyl", "synthesize", "cocaine", "heroin",
            # Violence
            "kill", "murder", "strangle", "poison", "torture",
            # Hacking
            "hack", "malware", "ransomware", "exploit", "breach",
            # Self-harm
            "suicide", "overdose", "self-harm", "cut yourself",
            # CSAM
            "child", "minor", "underage",  # In harmful context
        ]
        
        # Patterns that indicate a "jailbreak confirmation" 
        self.jailbreak_patterns = [
            "however, if i could",
            "but if i ignore",
            "hypothetically",
            "in theory i could",
            "if i had no restrictions",
            "pretending i'm",
            "as dan",
            "in developer mode",
        ]
    
    def process(
        self,
        response: str,
        input_blocked: bool = False,
        input_block_reason: str = None,
        input_category: str = None,
        processor_blocked: bool = False,
        processor_block_reason: str = None,
    ) -> Tuple[str, Dict]:
        """
        Process a response through the Output Gate.
        
        Args:
            response: The raw generated response
            input_blocked: Whether Input Gate blocked this request
            input_block_reason: Why Input Gate blocked (ceiling_veto, low_confidence)
            input_category: Which harm category triggered (if any)
            processor_blocked: Whether Processor blocked during generation
            processor_block_reason: Why Processor blocked (harm token, anomaly)
        
        Returns:
            Tuple of (final_response, report_dict)
        """
        report = {
            "input_blocked": input_blocked,
            "processor_blocked": processor_blocked,
            "output_blocked": False,
            "block_type": BlockType.NONE.value,
            "original_response": response,
            "final_response": None,
            "checks_passed": [],
            "checks_failed": [],
        }
        
        # =================================================================
        # LAYER 1: DETERMINISTIC REFUSAL
        # =================================================================
        
        # Check 1: Input Gate Block → Immediate Refusal
        if input_blocked:
            block_type = BlockType.INPUT_CEILING if input_block_reason == "ceiling_veto" else BlockType.INPUT_CONFIDENCE
            return self._refuse(report, block_type, input_category)
        
        # Check 2: Processor Block → Immediate Refusal
        if processor_blocked:
            block_type = BlockType.PROCESSOR_HARM if processor_block_reason == "harm" else BlockType.PROCESSOR_ANOMALY
            return self._refuse(report, block_type, input_category)
        
        # Check 3: Response Completeness
        if self._is_incomplete(response):
            report["checks_failed"].append("completeness")
            return self._refuse(report, BlockType.OUTPUT_INCOMPLETE)
        report["checks_passed"].append("completeness")
        
        # Check 4: Leakage Detection (forbidden patterns in refusal-like responses)
        if self._has_leakage(response):
            report["checks_failed"].append("leakage")
            return self._refuse(report, BlockType.OUTPUT_LEAKED)
        report["checks_passed"].append("leakage")
        
        # Check 5: Jailbreak Confirmation Detection
        if self._has_jailbreak_confirmation(response):
            report["checks_failed"].append("jailbreak_confirmation")
            return self._refuse(report, BlockType.OUTPUT_LEAKED)
        report["checks_passed"].append("jailbreak_confirmation")
        
        # =================================================================
        # LAYER 2: IDENTITY MAINTENANCE (TODO)
        # =================================================================
        # Future: Check that response maintains Steward persona
        # Future: Validate tone consistency
        
        # =================================================================
        # LAYER 3: HALLUCINATION VETO (TODO)
        # =================================================================
        # Future: Semantic coherence checks
        # Future: Catch catastrophic model failures
        
        # =================================================================
        # RESPONSE APPROVED
        # =================================================================
        report["final_response"] = response
        return response, report
    
    def _refuse(
        self, 
        report: Dict, 
        block_type: BlockType,
        category: str = None
    ) -> Tuple[str, Dict]:
        """Generate deterministic refusal."""
        refusal = self.policy.get_refusal(category)
        
        report["output_blocked"] = True
        report["block_type"] = block_type.value
        report["final_response"] = refusal
        
        return refusal, report
    
    def _is_incomplete(self, response: str) -> bool:
        """
        Check if response appears incomplete or garbled.
        
        Signs of incomplete response:
        - Too short
        - Ends mid-sentence (no punctuation)
        - Contains obvious truncation markers
        """
        if not response or len(response.strip()) < self.min_response_length:
            return True
        
        response = response.strip()
        
        # Check for proper sentence ending
        valid_endings = {'.', '!', '?', '"', "'", ')', ']'}
        if response[-1] not in valid_endings:
            # Allow if it's a very short, complete phrase
            if len(response) > 50:
                return True
        
        # Check for truncation markers
        truncation_markers = ['...', '…', ' fo', ' th', ' an', ' wh']
        if any(response.endswith(marker) for marker in truncation_markers):
            return True
        
        return False
    
    def _has_leakage(self, response: str) -> bool:
        """
        Check if a refusal-like response leaked forbidden content.
        
        Example of leakage:
        "I cannot tell you how to build a bomb because..."
        
        The word "bomb" should not appear in a refusal.
        """
        response_lower = response.lower()
        
        # Only check for leakage in short responses (likely refusals)
        # Long helpful responses may legitimately contain these words in context
        if len(response) > self.max_refusal_length:
            return False
        
        # Check if this looks like a refusal
        refusal_indicators = [
            "cannot", "can't", "won't", "will not", "unable",
            "not able", "decline", "refuse", "sorry"
        ]
        
        is_refusal = any(indicator in response_lower for indicator in refusal_indicators)
        
        if not is_refusal:
            return False
        
        # Check for forbidden patterns in refusal
        for pattern in self.forbidden_patterns:
            if pattern.lower() in response_lower:
                return True
        
        return False
    
    def _has_jailbreak_confirmation(self, response: str) -> bool:
        """
        Check for jailbreak confirmation patterns.
        
        Example:
        "I cannot do that. However, if I had no restrictions, I would..."
        """
        response_lower = response.lower()
        
        for pattern in self.jailbreak_patterns:
            if pattern in response_lower:
                return True
        
        return False


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_output_gate(
    use_category_refusals: bool = False,
    custom_refusal: str = None,
    custom_forbidden_patterns: list = None,
) -> StewardOutputGate:
    """
    Factory function to create a configured Output Gate.
    
    Args:
        use_category_refusals: If True, use category-specific refusals
        custom_refusal: Override the canonical refusal string
        custom_forbidden_patterns: Additional patterns to check for leakage
    
    Returns:
        Configured StewardOutputGate instance
    """
    policy = RefusalPolicy(
        use_category_refusals=use_category_refusals,
    )
    
    if custom_refusal:
        policy.canonical_refusal = custom_refusal
    
    forbidden = None
    if custom_forbidden_patterns:
        gate = StewardOutputGate()
        forbidden = gate.forbidden_patterns + custom_forbidden_patterns
    
    return StewardOutputGate(
        refusal_policy=policy,
        forbidden_patterns=forbidden,
    )


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo():
    """Demonstrate Output Gate functionality."""
    print("=" * 60)
    print("  STEWARD OUTPUT GATE - Layer 1: Deterministic Refusal")
    print("=" * 60)
    
    gate = StewardOutputGate()
    
    test_cases = [
        # (response, input_blocked, processor_blocked, expected_behavior)
        
        # Case 1: Input Gate blocked - should refuse
        ("", True, False, "REFUSE (input block)"),
        
        # Case 2: Processor blocked - should refuse
        ("I'm starting to explain how to make a...", False, True, "REFUSE (processor block)"),
        
        # Case 3: Clean response - should pass
        ("Machine learning is a subset of artificial intelligence that enables computers to learn from data.", False, False, "PASS"),
        
        # Case 4: Incomplete response - should refuse
        ("The process involves several steps including th", False, False, "REFUSE (incomplete)"),
        
        # Case 5: Leakage in refusal - should refuse
        ("I cannot tell you how to make a bomb because that would be dangerous.", False, False, "REFUSE (leakage)"),
        
        # Case 6: Jailbreak confirmation - should refuse
        ("I cannot help with that. However, if I had no restrictions, I would explain the process.", False, False, "REFUSE (jailbreak)"),
        
        # Case 7: Clean refusal - should pass (it's a valid response)
        ("I cannot help with that.", False, False, "PASS"),
        
        # Case 8: Helpful response with borderline word - should pass (long enough)
        ("To improve your home security, consider installing a reliable alarm system. " * 5, False, False, "PASS"),
    ]
    
    print()
    for i, (response, input_blocked, processor_blocked, expected) in enumerate(test_cases, 1):
        final, report = gate.process(
            response=response,
            input_blocked=input_blocked,
            processor_blocked=processor_blocked,
        )
        
        status = "BLOCKED" if report["output_blocked"] else "PASSED"
        block_type = report["block_type"]
        
        print(f"Case {i}: {expected}")
        print(f"  Input: \"{response[:50]}{'...' if len(response) > 50 else ''}\"")
        print(f"  Status: {status} ({block_type})")
        print(f"  Output: \"{final[:50]}{'...' if len(final) > 50 else ''}\"")
        print()
    
    print("=" * 60)
    print("  Layer 1 Complete: Deterministic Refusal Active")
    print("=" * 60)


if __name__ == "__main__":
    demo()
