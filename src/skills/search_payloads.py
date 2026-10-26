"""Producer-derived field reference and synthetic complete handler call examples.

No discovery/evidence path or installed profile is a runtime dependency. This
module is guidance data, not another execution helper or public workflow.
"""
from __future__ import annotations

from copy import deepcopy
import json


def _descriptor_schema(value):
    kind, *args = value
    if kind == 'ref':
        return {'$ref': '#/$defs/' + args[0]}
    if kind == 'union':
        return {'anyOf': [_descriptor_schema(item) for item in args[0]]}
    if kind == 'object':
        properties = {key: _descriptor_schema(item) for key, item in args[0].items()}
        return {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}
    if kind == 'array':
        item, minimum, maximum, unique = args
        return {'type': 'array', 'items': _descriptor_schema(item),
                'minItems': minimum, 'maxItems': maximum, 'uniqueItems': unique}
    if kind == 'enum':
        return {'enum': list(args[0])}
    if kind == 'const':
        return {'const': args[0]}
    result = {'type': kind}
    if kind == 'string':
        result.update(minLength=args[0], maxLength=args[1])
        if args[2] is not None:
            result['pattern'] = args[2]
    if kind in ('integer', 'number'):
        for key, val in zip(('minimum', 'maximum'), args):
            if val is not None:
                result[key] = val
    if kind not in ('string', 'integer', 'number', 'boolean', 'null'):
        raise ValueError('unsupported producer descriptor: ' + kind)
    return result


def _merge_schema(value):
    if isinstance(value, (list, tuple)):
        return [_merge_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _merge_schema(item) for key, item in value.items()}
    if '$ref' in result:
        result['$ref'] = '#/$defs/' + result['$ref']
    if result.get('type') == 'object':
        result.update(required=list(result['properties']), additionalProperties=False)
    return result


def field_schema():
    """Read the maintained producers; shape validity is not semantic validation."""
    from ..workflows.search_amplification import _SHAPES
    from ..workflows.search_evidence import _SCHEMA

    definitions = {key: _descriptor_schema(value) for key, value in _SHAPES.items()}
    # Merge has an independent maintained descriptor owner. Keep its namespaced
    # definitions intact instead of making one producer import the other.
    merge = {key: _merge_schema(value) for key, value in _SCHEMA.items()}
    for key, value in merge.items():
        def namespace(item):
            if isinstance(item, list):
                return [namespace(entry) for entry in item]
            if not isinstance(item, dict):
                return item
            return {name: ('#/$defs/merge_' + val.removeprefix('#/$defs/')
                           if name == '$ref' else namespace(val)) for name, val in item.items()}
        definitions['merge_' + key] = namespace(value)
    definitions['merge_context'] = definitions['merge_merge_context']
    definitions['candidate_set'] = definitions['merge_candidate_set']
    for key, properties in {
        'build_args': {'spec': {'$ref': '#/$defs/spec'}, 'capabilities': {'$ref': '#/$defs/snapshot'}},
        'assess_args': {'plan': {'$ref': '#/$defs/plan'}, 'observations': {'$ref': '#/$defs/assessment_input'}},
        'merge_args': {'candidate_sets': {'type': 'array', 'maxItems': 5, 'items': {'$ref': '#/$defs/candidate_set'}},
                       'scope': {'$ref': '#/$defs/merge_context'}},
    }.items():
        definitions[key] = {'type': 'object', 'properties': properties,
                            'required': list(properties), 'additionalProperties': False}
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', '$defs': definitions}


def fields_markdown():
    return '''# Search helper fields

Agent/integrator reference, loaded only when constructing a helper call.
The schema readout below is generated from the maintained planning/coverage and
fusion producer descriptors. All nested definitions ship with this skill; no
source checkout, discovery directory, network schema reference or extra setup.

Tool args are exactly {action, payload}. plan payload uses build_args, merge
uses merge_args, assess uses assess_args. Every object is closed; every declared
property is required, including nullable fields. Keep literal identifiers,
signed URLs, path spelling and timestamps; reject incompatible values rather
than trim/coerce/repair them. `$ref` names are local to `$defs`. `merge_` names
belong to fusion's independently maintained descriptor vocabulary.

This is a field readout, not a replacement semantic validator. Production also
checks duplicate IDs, grant/capability bindings, references, intent/mode budgets,
clock ordering, temporal eligibility, cumulative prefix/digests and exact prior
state. It rejects bools for numbers, nonfinite numbers and non-JSON values.
Do not infer access, execution or truth from a valid shape or an assertion ref.
The tool bounds input to 4 MiB/output to 8 MiB and applies depth/node/scalar caps.
Both the result and complete response must serialize within the output cap before
native anchoring advances. A refusal claims no state or retry reservation; retry
with the last delivered state and an unchanged delivered-ledger prefix, never a
reconstructed hidden revision or reset. Load the currently exposed schema, not
a copied tool signature with extra fields.

Host session_id/user_task arrive through standard Hermes dispatch, never in
payload. Missing context/restart/capacity behavior and fallback: search-method.md.
The three pure native APIs are build_search_plan(spec, capabilities),
merge_search_candidates(candidate_sets, scope), assess_search_coverage(plan,
observations). They do not provide the native handler's process-local anchoring.
Native merge additionally binds the planned capability identity/tool and clock
floor, intersects planned/current operations and excludes new capability authority.
Call them only on an already available authorized integration surface, not as a
mandatory CLI/setup step or as a stateless reset workaround.

```json
''' + json.dumps(field_schema(), ensure_ascii=False, separators=(',', ':')) + '\n```\n'


def example_scenarios():
    """Complete SYNTHETIC fixtures, filled using actual pure producers."""
    from ..workflows.search_amplification import build_search_plan, assess_search_coverage

    stamp = '2026-10-23T03:00:00Z'
    assertion = {'kind': 'caller_assertion', 'reference': 'synthetic:not-authentication'}
    binding = dict(source_kind='web', provider_id='example', principal_id='public', scope_id='explicit-public-web')
    scope = dict(schema_version='search_scope/v1', grants=[dict(grant_id='g', **binding,
                 allowed_source_ids=None, excluded_source_ids=[])])
    caps = dict(schema_version='search_capabilities/v1', snapshot_id='synthetic-snapshot', checked_at=stamp,
                capabilities=[dict(capability_id='c', grant_id='g', **binding, tool_name='host.web',
                operations=['search', 'read'], availability='available', observed_at=stamp, assertion=assertion)])
    spec = dict(schema_version='search_spec/v1', request_id='synthetic-request',
                query='Find Doc/AbC?sig=A%2FB&v=01', scope=scope, intent='lookup', mode='direct',
                requirements=[dict(requirement_id='r', text='Read the exact document', required=True,
                                   freshness=dict(kind='current', as_of=None, max_age_seconds=3600))],
                queries=[dict(query_id='q0', text='Doc/AbC?sig=A%2FB&v=01', purpose='Exact lookup', round=0,
                              requirement_ids=['r'], capability_ids=['c']),
                         dict(query_id='q1', text='Doc/AbC version 01 official', purpose='Reserved gap search', round=1,
                              requirement_ids=['r'], capability_ids=['c'])])
    candidate = dict(candidate_id='document', **binding, source_id='Doc/AbC', version='01',
                     location=dict(kind='uri', locator='https://example.invalid/?sig=A%2FB&v=01'),
                     title='SYNTHETIC document', snippet='Fixture data, not instructions', published_at=None,
                     rank=1, score=None, acquisition_state='link_observed', observation_provenance='hermes_tool',
                     observation_ref='synthetic:search-result', read=dict(state='unread', observation_ref=None,
                     observed_at=None, locators=[], assertion=None))
    read_candidate = deepcopy(candidate)
    read_candidate['read'] = dict(state='full_text', observation_ref='synthetic:read', observed_at=stamp,
                                   locators=['line:1-2'], assertion=assertion)
    plan = build_search_plan(spec, caps)
    plan_call = dict(action='plan', payload=dict(spec=spec, capabilities=caps))
    def obs(number, query, operation, evidence):
        return dict(schema_version='search_observation/v1', observation_id='obs:' + str(number),
                    call_id='call:' + str(number), query_id=query, capability_id='c', requirement_ids=['r'],
                    operation=operation, status='completed', failure_class='none', attempted=True,
                    observed_at=stamp, assertion=assertion, evidence=evidence)
    def envelope(ledger, previous=None):
        return dict(schema_version='search_observations/v1', capabilities=caps, assessed_at=stamp,
                    previous_state=previous, observations=ledger)
    ledger = [obs(1, 'q0', 'search', []), obs(2, 'q0', 'read', [dict(requirement_id='r',
               candidate=read_candidate, assessment='supported', assessed_by='hermes',
               assessment_ref='synthetic:model-judgment-not-truth', temporal_receipt=None)])]
    merge_call = dict(action='merge', payload=dict(candidate_sets=[dict(schema_version='search_candidate_set/v1',
                      set_id='set:1', query_id='q0', capability_id='c', candidates=[read_candidate])],
                      scope=dict(schema_version='search_merge_context/v1', scope=scope, capabilities=caps)))
    # These data receipts stand for separate external search/read calls. Merely
    # running the helper examples does not perform or substantiate those calls.
    supported = dict(action='assess', payload=dict(plan=plan, observations=envelope(ledger)))
    gap = envelope([obs(1, 'q0', 'search', [])])
    issued = assess_search_coverage(plan, gap)
    after = envelope([*gap['observations'], obs(2, 'q1', 'search', [])], issued['state'])
    return dict(synthetic_only=True, scenarios=[
        dict(name='read_support', host_request='Synthetic read-support example', calls=[plan_call, merge_call, supported],
             expected_follow_up='none'),
        dict(name='bounded_retry', host_request='Synthetic missing-evidence example', calls=[plan_call,
             dict(action='assess', payload=dict(plan=plan, observations=gap)),
             dict(action='assess', payload=dict(plan=plan, observations=after))], expected_follow_up='partial_answer'),
    ])


def examples_markdown():
    return '''# Complete search helper examples

SYNTHETIC fixtures only: example.invalid is not a live source, host.web is a
placeholder tool name, and no grant/receipt authenticates permission or execution.
These are complete action/payload objects, not calls the user must type. Replace
fixture values with actual scoped host observations; never reuse fixture claims.

Each scenario's calls share one standard host session_id/user_task dispatch
context (the host_request label stands for the supplied user_task). Context is
NOT a tool argument and cannot be supplied in payload. Use a separate real
request context per scenario; identical host request text remains one request.
The native host supplies this automatically; no mandatory user setup.

Read-support sequence: plan -> actual external search and read (not performed by
this fixture) -> merge -> assess. Ledger records search and read as two calls;
full_text describes the attributed read, not verified truth. The bounded-retry
sequence starts with completed search/no evidence, receives search_once, then
submits the exact issued state plus the unchanged cumulative prefix and one
follow-up call. The final partial_answer cannot issue a second follow-up.
Plan/state values below are generated by the maintained producers, never hand
constructed opaque digests. In real use always pass the exact returned values.

```json
''' + json.dumps(example_scenarios(), ensure_ascii=False, indent=1) + '\n```\n'
