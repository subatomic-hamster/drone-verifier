"""The tools agents read: name -> description. Pure data (no dvsim, no mcp) so the prompt can be rendered anywhere."""

TOOLS = {
    "task_info": "The mission, the limits, the gates, your remaining budget and the state of your design and controller. No arguments.",
    "parts_search": "The catalog: options with specs and price for a slot (battery, esc, servo, regulator, gnss, range, legs), or all slots if slot is omitted. query filters by name.",
    "design_set": "Choose parts: choices maps slot to part id, plus optional battery_x_mm and battery_y_mm (pack position on the deck). Slots you leave out keep their choice. Returns cost, mass, thrust to weight, centre of mass, and every broken constraint.",
    "design_report": "The current design in full: bill of materials, design rule findings, mass properties, static thrust, and `cfg`, the numbers your controller receives.",
    "write_controller": "Save your flight controller: code is one Python file with `class Controller`. Rejected with line numbers if it breaks the rules. If the design is complete it is also started once and stepped once, to catch crashes.",
    "fly": "Fly one scenario (nominal, wind or rain) for debugging: returns pass or fail, the first violation, metrics and telemetry at 2 Hz. Does not use a verify call and does not score.",
    "verify": "Run the gates in order on your current design and controller; stops at the first failed gate. Uses one verify call. Returns every gate's result and why it failed.",
    "finish": "Stop. note is a short summary. Your last design and controller are then flown on seeds you have not seen; that is your score. Nothing can be changed afterwards.",
}
