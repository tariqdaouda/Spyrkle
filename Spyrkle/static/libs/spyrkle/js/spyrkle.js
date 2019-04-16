var toggle_page = function(page_id){
    $("spyrkle-page").hide()
    $("#" + page_id).show()
    $("spyrkle-page-selector").removeClass('uk-button-primary')
    $("#" + page_id + '-' + 'selector').addClass('uk-button-primary')
} 